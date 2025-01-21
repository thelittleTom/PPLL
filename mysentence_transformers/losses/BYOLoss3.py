import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from ..SentenceTransformer import SentenceTransformer
import torch
import torch.nn as nn
import numpy as np
import logging
import math
from functools import wraps
import copy
import random


def batch_to_device(batch, target_device):
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


# MLP for  predictor
class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


# loss fn
def loss_fn(x, y, l):
    temperature = 1
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    sim = (x * y / temperature).sum(dim=-1)
    # return -(l*torch.log(sim)+(1-l)*torch.log(1-sim))
    return 2 - 2 * (x * y / temperature).sum(dim=-1)


# return l*(2 - 2 * (x * y / temperature).sum(dim=-1))
# return (l-(x * y/temperature).sum(dim=-1))**2

# 实现求出两两之间的相似性
# 假设 x 和 y 是 (n, d) 的张量
# x = torch.tensor([[3.0, 4.0], [1.0, 2.0], [2.0, 3.0]])  # 3个d=2维的向量
# y = torch.tensor([[5.0, 12.0], [0.0, 1.0], [1.0, 1.0]])
#
# # 归一化
# x = F.normalize(x, dim=-1, p=2)
# y = F.normalize(y, dim=-1, p=2)
#
# # 计算两两之间的余弦相似度
# cos_sim_matrix = torch.mm(x, y.T)  # 点积等价于余弦相似度，因为已经归一化
# print(cos_sim_matrix)


class BYOLoss3(nn.Module):
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 moving_average_decay: float):
        super(BYOLoss3, self).__init__()
        self.online_encoder = model
        self.online_predictor_1 = MLP(sentence_embedding_dimension, sentence_embedding_dimension,
                                      10 * sentence_embedding_dimension)
        self.online_predictor_2 = MLP(sentence_embedding_dimension, sentence_embedding_dimension,
                                      10 * sentence_embedding_dimension)
        self.online_predictor_3 = MLP(sentence_embedding_dimension, sentence_embedding_dimension,
                                      10 * sentence_embedding_dimension)
        self.target_encoder = copy.deepcopy(self.online_encoder)

        model_pth = f'/root/autodl-tmp/paper_cluster/output/f/0-0.0001-1-12-17_20-41/checkpoint/0_600.pth'
        state_dict = torch.load(model_pth)
        self.target_encoder.load_state_dict(state_dict)

        self.target_ema_updater = EMA(moving_average_decay)
        self.eps = 1e-7
    def update_moving_average(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
    def mle(self, teacher_top1_sim_pred, student_top1_sim_pred):

        y_pred = student_top1_sim_pred
        y_true = teacher_top1_sim_pred

        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)
        mask = y_true_sorted == -1
        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float('-inf')
        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)
        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values
        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
        observation_loss = torch.log(cumsums + self.eps) - preds_sorted_by_true_minus_max
        observation_loss[mask] = 0.0

        return   torch.mean(torch.sum(observation_loss, dim=1))
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, device, llm_features, mask):
        target_sentence_features = copy.deepcopy(sentence_features)
        labels = labels.to(device)
        for s in sentence_features:
            batch_to_device(s, device)
        rep_one = self.online_encoder(sentence_features[0])

        for s in target_sentence_features:
            batch_to_device(s, device)
        with torch.no_grad():
            target_one = self.target_encoder(target_sentence_features[0])

        mode=1
        if llm_features is not None and mode==0:
            # rank loss
            for s in llm_features:
                batch_to_device(s, device)
            target_temp = 1 / 20
            online_temp = 1 / 10
            with torch.no_grad():
                # target_one, target_two = [self.target_encoder(feature) for feature in llm_features]
                target_proj_one = target_one['sentence_embedding']

                x = F.normalize(target_proj_one, dim=-1, p=2)
                cos_sim_matrix = torch.mm(x, x.T)  # 点积等价于余弦相似度，因为已经归一化
                # cos_sim_matrix[mask == 1] = 1
                # cos_sim_matrix[mask == -1] =0
                target_cos_sim = cos_sim_matrix / target_temp
                target_softmax = F.softmax(target_cos_sim.fill_diagonal_(float('-inf')), dim=-1)

            online_pred_one = rep_one['sentence_embedding']
            # online_pred_one = self.online_predictor_1(online_pred_one)
            # online_pred_one = self.online_predictor_2(online_pred_one)
            # online_pred_one = self.online_predictor_3(online_pred_one)
            x = F.normalize(online_pred_one, dim=-1, p=2)
            cos_sim_matrix  = torch.mm(x, x.T)  # 点积等价于余弦相似度，因为已经归一化
            # cos_sim_matrix[mask == 1] = 1
            online_cos_sim = cos_sim_matrix / online_temp
            online_log_softmax = F.log_softmax(online_cos_sim.fill_diagonal_(float('-inf')), dim=-1)
            loss2 = -(target_softmax * online_log_softmax).nansum() / target_softmax.nansum()
            return loss2
        elif llm_features is not None and mode==1:
            for s in llm_features:
                batch_to_device(s, device)
            target_temp = 1 / 20
            online_temp = 1 / 10
            with torch.no_grad():
                # target_one, target_two = [self.target_encoder(feature) for feature in llm_features]
                target_proj_one = target_one['sentence_embedding']

                x = F.normalize(target_proj_one, dim=-1, p=2)
                cos_sim_matrix = torch.mm(x, x.T)  # 点积等价于余弦相似度，因为已经归一化
                cos_sim_matrix[mask == 1] = 1
                # cos_sim_matrix[mask == -1] =0
                target_cos_sim = cos_sim_matrix / target_temp


            online_pred_one = rep_one['sentence_embedding']
            # online_pred_one = self.online_predictor_1(online_pred_one)
            # online_pred_one = self.online_predictor_2(online_pred_one)
            # online_pred_one = self.online_predictor_3(online_pred_one)
            x = F.normalize(online_pred_one, dim=-1, p=2)
            cos_sim_matrix  = torch.mm(x, x.T)  # 点积等价于余弦相似度，因为已经归一化
            # cos_sim_matrix[mask == 1] = 1
            online_cos_sim = cos_sim_matrix / online_temp
            return self.mle(target_cos_sim,online_cos_sim)
        # loss
