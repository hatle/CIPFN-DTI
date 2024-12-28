import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from dgllife.model.gnn import GCN
from torch.nn.utils.weight_norm import weight_norm

from Feature_enhancer import BiAttentionBlock
from glfusion import VisualAwarePromptingModule
from einops import reduce

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))   # 负对数似然损失
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss

def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1) -5
    return loss_ent

class CIPFNDTI(nn.Module):
    def __init__(self, **config):
        super(CIPFNDTI, self).__init__()

        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]     # 75
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]    # 128
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]    # [128,128,128]
        drug_padding = config["DRUG"]["PADDING"]              # True

        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]    # 128
        num_filters = config["PROTEIN"]["NUM_FILTERS"]          # [128,128,128]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]          # [3,6,9]
        protein_padding = config["PROTEIN"]["PADDING"]          # True

        mlp_in_dim = config["DECODER"]["IN_DIM"]                # 256
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]        # 512
        mlp_out_dim = config["DECODER"]["OUT_DIM"]              # 128
        out_binary = config["DECODER"]["BINARY"]                # 1

        ban_heads = config["BCN"]["HEADS"]                      # 2                               #


        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)

        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)

        self.attn_drug = MultiHeadSelfAttention(drug_embedding,4,drug_hidden_feats[-1],drug_hidden_feats[-1])
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

        # Gated Attention Unit
        self.gau = GAU(protein_emb_dim,num_filters[-1])

        # self.trans_protein = transformer()
        self.feature_enhancer = BiAttentionBlock(drug_hidden_feats[-1],num_filters[-1],mlp_in_dim,ban_heads)

        # interaction-aware
        self.gl = VisualAwarePromptingModule(128,128,128,mlp_in_dim,8)
        self.avg = nn.AvgPool1d(1)
        self.drn = DimensionalityReductionNetwork()

    def forward(self, bg_d, v_p,protein_mask, mode="train"):

        v_d = self.drug_extractor(bg_d)   # [8,290,128]-> [5,290,128]
        v_p = self.protein_extractor(v_p)   # [8,1185,128]->[5,1185,128]

        v_p = self.gau(v_p)
        v_d = self.attn_drug(v_d)

        attmask2 = torch.tril(torch.ones(protein_mask.size(0), 1185)).bool().to("cuda")
        # attmask1 = torch.tril(torch.ones(protein_mask.size(0), 290)).bool().to("cuda")
        v_d,v_p = self.feature_enhancer(v_d,v_p,None,attmask2)

        ff = self.gl(v_d,v_p)
        ff = self.avg(ff)
        ff = self.drn(ff,v_d,v_p)
        # ff = torch.mean(ff,dim=2)
        f = ff
        a = v_d
        b = v_p
        d_att = a.transpose(1, 2)
        p_att = b.transpose(1, 2)
        att = torch.cat((d_att, p_att), dim=-1)  # [8,128,1475]

        score = self.mlp_classifier(f)

        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att

class GAU(nn.Module):
    def __init__(
        self,
        dim,
        query_key_dim = 128,
        expansion_factor = 2.,
        add_residual = True,
        dropout = 0.,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_hidden = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.SiLU()
        )
        self.to_qk = nn.Sequential(
            nn.Linear(dim, query_key_dim),
            nn.SiLU()
        )
        self.gamma = nn.Parameter(torch.ones(2, query_key_dim))
        self.beta = nn.Parameter(torch.zeros(2, query_key_dim))
        nn.init.normal_(self.gamma, std=0.02)
        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

        self.add_residual = add_residual

    def forward(self, x):
        seq_len = x.shape[-2]
        normed_x = self.norm(x) #(bs,seq_len,dim)
        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1) #(bs,seq_len,seq_len)
        Z = self.to_qk(normed_x) #(bs,seq_len,query_key_dim)
        QK = torch.einsum('... d, h d -> ... h d', Z, self.gamma) + self.beta
        q, k = QK.unbind(dim=-2)
        sim = torch.einsum('b i d, b j d -> b i j', q, k) / seq_len
        A = F.relu(sim) ** 2
        A = self.dropout(A)
        V = torch.einsum('b i j, b j d -> b i d', A, v)
        V = V * gate
        out = self.to_out(V)
        if self.add_residual:
            out = out + x
        return out

# GCN模型
class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)

        return node_feats

# CNN模型
class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)  # 我设置26个索引，每个使用 embedding_dim个维度表达
        else:
            self.embedding = nn.Embedding(26, embedding_dim)

        in_ch = [embedding_dim] + num_filters  # [128,128,128,128]
        self.in_ch = in_ch[-1]
        kernels = kernel_size

        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)
        return v

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, key_size, value_size, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_head_dim = key_size // num_heads
        self.k_head_dim = key_size // num_heads
        self.v_head_dim = value_size // num_heads

        self.W_q = nn.Linear(embed_dim, key_size, bias=bias)

        self.W_k = nn.Linear(embed_dim, key_size, bias=bias)
        self.W_v = nn.Linear(embed_dim, value_size, bias=bias)

        self.q_proj = nn.Linear(key_size, key_size, bias=bias)
        self.k_proj = nn.Linear(key_size, key_size, bias=bias)
        self.v_proj = nn.Linear(value_size, value_size, bias=bias)
        self.out_proj = nn.Linear(value_size, embed_dim, bias=bias)


    def forward(self, x):
        """
        Args:
            X: shape: (N, L, embed_dim), input sequence,
        Returns:
            output: (N, L, embed_dim)
        """
        query = self.W_q(x)  # (N, L, key_size)
        key = self.W_k(x)  # (N, L, key_size)
        value = self.W_v(x)  # (N, L, value_size)
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
        N, L, value_size = v.size()

        q = q.reshape(N, L, self.num_heads, self.q_head_dim).transpose(1, 2)
        k = k.reshape(N, L, self.num_heads, self.k_head_dim).transpose(1, 2)
        v = v.reshape(N, L, self.num_heads, self.v_head_dim).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.size(-1))
        att = F.softmax(att, dim=-1)
        output = torch.matmul(att, v)
        output = output.transpose(1, 2).reshape(N, L, value_size)
        output = self.out_proj(output)
        return output

# MLPDecoder
class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        # if binary:
        #     self.fc4 = nn.Linear(out_dim, binary)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))

        # if hasattr(self, 'fc4'):
        #     x = self.fc4(x)
        x = self.fc4(x)
        return x

# 分类器
class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list, output_dim=256):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

class DimensionalityReductionNetwork(nn.Module):
    def __init__(self):
        super(DimensionalityReductionNetwork, self).__init__()
        # Define dimensionality reduction layers for each input tensor
        self.reduce1 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.reduce2 = nn.Conv1d(in_channels=290, out_channels=128, kernel_size=1)
        self.reduce3 = nn.Conv1d(in_channels=1185, out_channels=128, kernel_size=1)

        # Define layers for processing the concatenated features
        self.conv1 = nn.Conv1d(in_channels=3 * 128, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1)
        self.fc = nn.Linear(128 * 128, 256)  # Final output is [8, 256]
        self.dropout = nn.Dropout(0.1)

    def forward(self, x1, x2, x3):
        # Apply dimensionality reduction
        x1 = F.relu(self.reduce1(x1))
        x2 = F.relu(self.reduce2(x2))
        x3 = F.relu(self.reduce3(x3))

        # Concatenate along the feature dimension
        x = torch.cat((x1, x2, x3), dim=1)  # x shape will be [8, 3*128, 128]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
        x = self.fc(x)

        # v_d,v_p
        # x2 = reduce(x2, 'B H W -> B W', 'max')
        # x3 = reduce(x3, 'B H W -> B W', 'max')
        #

        return x
