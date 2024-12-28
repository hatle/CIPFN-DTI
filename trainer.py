import torch
import torch.nn as nn
import pandas as pd
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from models import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer
from prettytable import PrettyTable
from domain_adaptator import ReverseLayerF
from tqdm import tqdm



class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, opt_da=None, discriminator=None,
                 experiment=None, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.is_da = config["DA"]["USE"]
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]

        if opt_da:
            self.optim_da = opt_da

        # if self.is_da:
        #
        #     self.da_method = config["DA"]["METHOD"]
        #     self.domain_dmm = discriminator
        #
        #     if config["DA"]["RANDOM_LAYER"] and not config["DA"]["ORIGINAL_RANDOM"]:
        #
        #         self.random_layer = nn.Linear(in_features=config["DECODER"]["IN_DIM"]*self.n_class, out_features=config["DA"]
        #         ["RANDOM_DIM"], bias=False).to(self.device)
        #
        #         torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)
        #         for param in self.random_layer.parameters():
        #
        #             param.requires_grad = False
        #     elif config["DA"]["RANDOM_LAYER"] and config["DA"]["ORIGINAL_RANDOM"]:
        #         self.random_layer = RandomLayer([config["DECODER"]["IN_DIM"], self.n_class], config["DA"]["RANDOM_DIM"])
        #         if torch.cuda.is_available():
        #             self.random_layer.cuda()
        #     else:
        #         self.random_layer = False

        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"]["LAMB_DA"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.use_da_entropy = config["DA"]["USE_ENTROPY"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.experiment = experiment

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]
        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold", "Test_loss"]
        if not self.is_da:
            train_metric_header = ["# Epoch", "Train_loss"]
        else:
            train_metric_header = ["# Epoch", "Train_loss", "Model_loss", "epoch_lamb_da", "da_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)
        self.original_random = config["DA"]["ORIGINAL_RANDOM"]

    def da_lambda_decay(self):
        delta_epoch = self.current_epoch - self.da_init_epoch
        non_init_epoch = self.epochs - self.da_init_epoch
        p = (self.current_epoch + delta_epoch * self.nb_training) / (
                non_init_epoch * self.nb_training
        )
        grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        return self.init_lamb_da * grow_fact

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            if not self.is_da:
                train_loss = self.train_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
                if self.experiment:  # 检查是否存在 Comet 实验对象
                    # 将总体损失记录到 Comet 实验中
                    self.experiment.log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)
            # else:
            #     train_loss, model_loss, da_loss, epoch_lamb = self.train_da_epoch()
            #     train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss, model_loss,
            #                                                                             epoch_lamb, da_loss]))
            #     self.train_model_loss_epoch.append(model_loss)
            #     self.train_da_loss_epoch.append(da_loss)
            #     if self.experiment:
            #         self.experiment.log_metric("train_epoch total loss", train_loss, epoch=self.current_epoch)
            #         self.experiment.log_metric("train_epoch model loss", model_loss, epoch=self.current_epoch)
            #         if self.current_epoch >= self.da_init_epoch:
            #             self.experiment.log_metric("train_epoch da loss", da_loss, epoch=self.current_epoch)
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc, val_loss = self.test(dataloader="val")
            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_loss, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auroc", auroc, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auprc", auprc, epoch=self.current_epoch)
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)

            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc))

        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                   accuracy, thred_optim, test_loss]))
        # 进行测试
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " Thred_optim " + str(thred_optim))
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision
        self.save_result()
        if self.experiment:
            self.experiment.log_metric("valid_best_auroc", self.best_auroc)
            self.experiment.log_metric("valid_best_epoch", self.best_epoch)
            self.experiment.log_metric("test_auroc", self.test_metrics["auroc"])
            self.experiment.log_metric("test_auprc", self.test_metrics["auprc"])
            self.experiment.log_metric("test_sensitivity", self.test_metrics["sensitivity"])
            self.experiment.log_metric("test_specificity", self.test_metrics["specificity"])
            self.experiment.log_metric("test_accuracy", self.test_metrics["accuracy"])
            self.experiment.log_metric("test_threshold", self.test_metrics["thred_optim"])
            self.experiment.log_metric("test_f1", self.test_metrics["F1"])
            self.experiment.log_metric("test_precision", self.test_metrics["Precision"])
        return self.test_metrics

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        # if self.is_da:
        #     state["train_model_loss"] = self.train_model_loss_epoch
        #     state["train_da_loss"] = self.train_da_loss_epoch
        #     state["da_init_epoch"] = self.da_init_epoch
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w


    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d, v_p,protein_mask, labels) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            # v_d, v_p, labels = v_d.to(self.device), v_p.to(self.device), labels.float().to(self.device)
            v_d, v_p, protein_mask, labels = v_d.to(self.device), v_p.to(self.device),protein_mask.to(self.device), labels.float().to(self.device)
            self.optim.zero_grad() # 将模型的梯度归零，以防止梯度累积。
            v_d, v_p, f, score = self.model(v_d, v_p,protein_mask)
            if self.n_class == 1:
                n, loss = binary_cross_entropy(score, labels)
            else:
                n, loss = cross_entropy_logits(score, labels)
            # loss, n = combined_loss(v_d,v_p,labels,score,self.n_class)

            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)
        loss_epoch = loss_epoch / num_batches  #  计算整个 epoch 的平均损失。
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch

    # def train_da_epoch(self):
    #     self.model.train()
    #     total_loss_epoch = 0
    #     model_loss_epoch = 0
    #     da_loss_epoch = 0
    #     # 用于存储领域自适应损失的权重
    #     epoch_lamb_da = 0
    #     if self.current_epoch >= self.da_init_epoch:
    #         # epoch_lamb_da = self.da_lambda_decay()
    #         epoch_lamb_da = 1
    #         if self.experiment:
    #             self.experiment.log_metric("DA loss lambda", epoch_lamb_da, epoch=self.current_epoch)
    #     #   获取训练数据集的批次数量
    #     num_batches = len(self.train_dataloader)
    #     for i, (batch_s, batch_t) in enumerate(tqdm(self.train_dataloader)):
    #         # 这是用于跟踪训练过程中的步数或迭代次数。
    #         self.step += 1
    #         # 从源域中获取数据 v_d 和 v_p 是输入数据，labels 是对应的标签。
    #         v_d, v_p, p_mask, labels = batch_s[0].to(self.device), batch_s[1].to(self.device),batch_s[2].to(self.device), batch_s[3].float().to(
    #             self.device)
    #
    #         # 用于域自适应训练
    #         v_d_t, v_p_t = batch_t[0].to(self.device), batch_t[1].to(self.device)
    #         # 将模型参数的梯度归零，以便进行新一轮的反向传播和梯度更新。
    #         # self.optim 和 self.optim_da 可能是用于优化的优化器，分别用于源域任务和领域自适应任务。
    #         self.optim.zero_grad()
    #         self.optim_da.zero_grad()
    #         # 调用模型 (self.model) 来进行前向传播，得到模型的输出 score
    #         v_d, v_p, f, score = self.model(v_d, v_p, p_mask)
    #         # 类别数为1
    #         if self.n_class == 1:
    #             n, model_loss = binary_cross_entropy(score, labels)  # 使用二进制交叉熵损失函数
    #         else:
    #             n, model_loss = cross_entropy_logits(score, labels)  # 使用对数软最大值损失函数
    #
    #         if self.current_epoch >= self.da_init_epoch:
    #             # 这行代码对目标域 (v_d_t, v_p_t) 进行前向传播，得到模型的输出 t_score。f_t 可能是中间特征，用于领域自适应的计算
    #             v_d_t, v_p_t, f_t, t_score = self.model(v_d_t, v_p_t, p_mask)
    #             # 使用 CDAN 方法计算源域和目标域的领域自适应损失。
    #             if self.da_method == "CDAN":
    #                 # 使用 ReverseLayerF 对源域的中间特征 f 进行反向传播，并根据超参数 self.alpha 控制反向传播的权重
    #                 reverse_f = ReverseLayerF.apply(f, self.alpha)
    #
    #                 softmax_output = torch.nn.Softmax(dim=1)(score)
    #                 softmax_output = softmax_output.detach()
    #                 # reverse_output = ReverseLayerF.apply(softmax_output, self.alpha)
    #                 if self.original_random:
    #                     # 通过随机层 (self.random_layer) 对反向传播的源域特征 (reverse_f) 和 softmax 输出 (softmax_output) 进行处理，
    #                     # 然后通过领域判别器 (self.domain_dmm) 得到对抗性输出的分数 (adv_output_src_score)
    #                     random_out = self.random_layer.forward([reverse_f, softmax_output])
    #                     adv_output_src_score = self.domain_dmm(random_out.view(-1, random_out.size(1)))
    #                 else:
    #                     # 不使用原始的随机层，通过源域的 softmax 输出和反向传播的特征计算特征矩阵 (feature)
    #                     feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
    #                     feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
    #                     if self.random_layer:
    #                         random_out = self.random_layer.forward(feature)
    #                         adv_output_src_score = self.domain_dmm(random_out)
    #                     else:
    #                         adv_output_src_score = self.domain_dmm(feature)
    #                 # ReverseLayerF.apply 是一个用于反向传播中的函数，它会对输入的特征 f_t 进行反向梯度传播，
    #                 # 但是在计算梯度时，会乘以一个负的权重 self.alpha，从而实现对特征的反向传播。
    #                 reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)
    #                 # Softmax 操作将分数转换为概率分布，使得每个类别的概率值都在 (0, 1) 的范围内，且所有类别的概率之和为 1。
    #                 softmax_output_t = torch.nn.Softmax(dim=1)(t_score)
    #                 # 将目标域的 softmax 输出 softmax_output_t 从计算图中分离出来，使其不再参与梯度计算。
    #                 # 这通常用于防止梯度的传播到不希望更新的部分，以避免对目标域产生不良影响。
    #                 softmax_output_t = softmax_output_t.detach()
    #                 # reverse_output_t = ReverseLayerF.apply(softmax_output_t, self.alpha)
    #                 # 如果使用了 original_random，则会调用 self.random_layer.forward 方法，
    #                 # 将目标域的反向传播特征 reverse_f_t 和经过 softmax 处理后的输出 softmax_output_t 传递给 self.ra
    #                 if self.original_random:
    #                     random_out_t = self.random_layer.forward([reverse_f_t, softmax_output_t])
    #                     adv_output_tgt_score = self.domain_dmm(random_out_t.view(-1, random_out_t.size(1)))
    #                 # 如果没有使用 original_random，则会通过计算目标域的特征 feature_t，该特征是目标域 softmax 输出和反向传播特征的叉乘。
    #                 # 然后，根据是否有 self.random_layer，决定是否对特征进行额外的随机化处理，最终得到目标域的领域自适应分数 adv_output_tgt_score。
    #                 # 这一过程是领域自适应方法中用于对抗域差异的一部分。
    #                 else:
    #                     # 这一行计算目标域的特征。它使用 PyTorch 的 bmm 函数，该函数执行批次矩阵乘法，
    #                     # 将目标域 softmax 输出 softmax_output_t 和反向传播特征 reverse_f_t 进行叉乘
    #                     feature_t = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
    #                     # 这一行将计算得到的特征展平成一维向量。这个向量将作为输入传递给后续的处理。
    #                     feature_t = feature_t.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
    #                     # 检查是否启用了随机层
    #                     if self.random_layer:
    #                         # 调用 self.random_layer.forward(feature_t) 对特征进行额外的随机化处理。
    #                         random_out_t = self.random_layer.forward(feature_t)
    #                         adv_output_tgt_score = self.domain_dmm(random_out_t)
    #                     else:
    #                         # 直接将特征传递给 self.domain_dmm，该函数用于计算目标域的领域自适应分数
    #                         adv_output_tgt_score = self.domain_dmm(feature_t)
    #
    #                 if self.use_da_entropy:
    #                     # 计算源域和目标域的熵权重
    #                     entropy_src = self._compute_entropy_weights(score)
    #                     entropy_tgt = self._compute_entropy_weights(t_score)
    #                     # 根据熵权重进行归一化，得到权重向量
    #                     src_weight = entropy_src / torch.sum(entropy_src)
    #                     tgt_weight = entropy_tgt / torch.sum(entropy_tgt)
    #                 else:
    #                     # 如果不使用熵权重，则将权重向量设置为 None
    #                     src_weight = None
    #                     tgt_weight = None
    #                 # 使用 cross_entropy_logits 函数计算源域和目标域的 CDAN 损失
    #                 n_src, loss_cdan_src = cross_entropy_logits(adv_output_src_score, torch.zeros(self.batch_size).to(self.device),
    #                                                             src_weight)
    #                 n_tgt, loss_cdan_tgt = cross_entropy_logits(adv_output_tgt_score, torch.ones(self.batch_size).to(self.device),
    #                                                             tgt_weight)
    #                 # 计算领域自适应损失，为源域和目标域的 CDAN 损失之和
    #                 da_loss = loss_cdan_src + loss_cdan_tgt
    #             else:
    #                 raise ValueError(f"The da method {self.da_method} is not supported")
    #             # 将模型损失和领域自适应损失相加得到总损失
    #             loss = model_loss + da_loss
    #         else:
    #             loss = model_loss
    #         # 反向传播和优化步骤
    #         loss.backward()
    #         self.optim.step()
    #         self.optim_da.step()
    #         # 累积总体损失、模型损失和领域自适应损失
    #         total_loss_epoch += loss.item()
    #         model_loss_epoch += model_loss.item()
    #         # 如果启用了实验记录（experiment），则记录模型损失和总体损失
    #         if self.experiment:
    #             self.experiment.log_metric("train_step model loss", model_loss.item(), step=self.step)
    #             self.experiment.log_metric("train_step total loss", loss.item(), step=self.step)
    #         # 如果当前 epoch 大于等于领域自适应的起始 epoch（da_init_epoch）
    #         if self.current_epoch >= self.da_init_epoch:
    #             # 累积领域自适应损失
    #             da_loss_epoch += da_loss.item()
    #             # 如果启用了实验记录，记录领域自适应损失
    #             if self.experiment:
    #                 self.experiment.log_metric("train_step da loss", da_loss.item(), step=self.step)
    #     # 返回平均损失
    #     total_loss_epoch = total_loss_epoch / num_batches
    #     model_loss_epoch = model_loss_epoch / num_batches
    #     da_loss_epoch = da_loss_epoch / num_batches
    #     # 如果当前 epoch 小于领域自适应的起始 epoch（da_init_epoch）
    #     if self.current_epoch < self.da_init_epoch:
    #         # 打印模型训练损失
    #         print('Training at Epoch ' + str(self.current_epoch) + ' with model training loss ' + str(total_loss_epoch))
    #     else:
    #         # 打印模型训练损失、领域自适应损失、总体训练损失和领域自适应权重（DA lambda）
    #         print('Training at Epoch ' + str(self.current_epoch) + ' model training loss ' + str(model_loss_epoch)
    #               + ", da loss " + str(da_loss_epoch) + ", total training loss " + str(total_loss_epoch) + ", DA lambda " +
    #               str(epoch_lamb_da))
    #     # 返回平均损失、模型损失、领域自适应损失和领域自适应权重（DA lambda）
    #     return total_loss_epoch, model_loss_epoch, da_loss_epoch, epoch_lamb_da

    def test(self, dataloader="test"):

        test_loss = 0
        y_label, y_pred = [], []

        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")

        num_batches = len(data_loader)
        # #
        # df = {'drug': [], 'protein': [], 'y_pred': [], 'y_label': []}
        # #

        with torch.no_grad():
            self.model.eval()
            for i, (v_d, v_p,p_mask, labels) in enumerate(data_loader):
                v_d, v_p,p_mask, labels = v_d.to(self.device), v_p.to(self.device),p_mask.to(self.device), labels.float().to(self.device)
                if dataloader == "val":
                    v_d, v_p, f, score = self.model(v_d, v_p,p_mask)
                elif dataloader == "test":
                    v_d, v_p, f, score = self.best_model(v_d, v_p,p_mask)
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                # loss, n = combined_loss(v_d, v_p, labels, score, self.n_class)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
                # #
                # if dataloader == 'test':
                #     df['drug'] = df['drug'] + v_d.to('cpu').tolist()
                #     df['protein'] = df['protein'] + v_p.to('cpu').tolist()
                # #
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches

        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

            if self.experiment:
                self.experiment.log_curve("test_roc curve", fpr, tpr)
                self.experiment.log_curve("test_pr curve", recall, prec)

            precision1 = precision_score(y_label, y_pred_s)
            # #
            # df['y_label'] = y_label
            # df['y_pred'] = y_pred
            # data = pd.DataFrame(df)
            # data.to_csv('result/visualization.csv', index=False)
            # #

            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1
        else:
            return auroc, auprc, test_loss
