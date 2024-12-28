import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class Prompt(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, num_heads=8, dropout=0.1):
        super(Prompt, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        layers = []
        current_dim = input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim

        # Final layer without LayerNorm and ReLU, as it directly projects to hidden_dim
        layers.append(nn.Linear(current_dim, hidden_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        attn_output, _ = self.self_attention(x, x, x)
        x = x + attn_output  # Add residual connection
        return x


class GatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(GatedFusion, self).__init__()
        self.Ws = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, Fs, Ev):
        K = V = self.ffn(Ev)
        Q = Fs
        Fattn_v = Q + torch.matmul(F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5), dim=-1), V)
        lambda_ = self.sigmoid(self.Ws(Fs) + self.Wv(Fattn_v))
        Fm = (1 - lambda_) * Fs + lambda_ * Fattn_v
        return Fm

class Resampler(nn.Module):
    def __init__(self, hidden_dim, k, num_heads):
        super(Resampler, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.k = k

    def forward(self, Fm):
        Fk = nn.Parameter(torch.Tensor(Fm.size(0), self.k, Fm.size(-1)).normal_()).to(device)

        attn_output, _ = self.cross_attention(Fk, Fm, Fm)
        Fk = attn_output + Fk
        Fk = self.ffn(Fk) + Fk
        return Fk

class VisualAwarePromptingModule(nn.Module):
    def __init__(self, prompt_input_dim, hidden_dim, visual_input_dim, k, num_heads):
        super(VisualAwarePromptingModule, self).__init__()
        self.prompt = Prompt(prompt_input_dim, hidden_dim)
        self.gated_fusion = GatedFusion(hidden_dim)
        self.resampler = Resampler(hidden_dim, k, num_heads)

    def forward(self, S, Ev):
        Fs = self.prompt(S)
        Fm = self.gated_fusion(Fs, Ev)
        Fk = self.resampler(Fm)
        return Fk


