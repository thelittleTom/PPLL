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

def batch_to_device(batch, target_device ):
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
def loss_fn(x, y,l):
	temperature =1
	x = F.normalize(x, dim=-1, p=2)
	y = F.normalize(y, dim=-1, p=2)
	dis=( 2 - 2 * (x * y/temperature).sum(dim=-1))
	return torch.max(dis - 0.03, torch.tensor(0.0))
	# sim=(x * y/temperature).sum(dim=-1)
	# return -(l*torch.log(sim)+(1-l)*torch.log(1-sim))
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
	# tensor([[0.9692, 0.8000, 0.9899],
	# 		[0.9976, 0.8944, 0.9487],
	# 		[0.9814, 0.8321, 0.9806]])


class BYOLoss4(nn.Module):
	def __init__(self,
				 model: SentenceTransformer,
				 sentence_embedding_dimension: int,
				 moving_average_decay: float):
		super(BYOLoss4, self).__init__()
		self.online_encoder = model
		intent_dimension=128
		# self.online_predictor_1 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension)
		# self.online_predictor_2 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension)
		# self.online_predictor_3 = MLP(sentence_embedding_dimension, sentence_embedding_dimension, 10 * sentence_embedding_dimension)
		# self.target_encoder = copy.deepcopy(self.online_encoder)
		self.online_predictor_1 = MLP(sentence_embedding_dimension, intent_dimension, 10 * sentence_embedding_dimension)
		self.online_predictor_2 = MLP(intent_dimension, intent_dimension, 10 * intent_dimension)
		self.online_predictor_3 = MLP(intent_dimension, intent_dimension, 10 * intent_dimension)
		self.target_encoder = copy.deepcopy(self.online_encoder)
		self.target_predictor_1=copy.deepcopy(self.online_predictor_1)




		self.target_ema_updater = EMA(moving_average_decay)

	def update_moving_average(self):
		assert self.target_encoder is not None, 'target encoder has not been created yet'
		update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)
		update_moving_average(self.target_ema_updater, self.target_predictor_1,self.online_predictor_1)


	def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor,device):
		target_sentence_features = copy.deepcopy(sentence_features)
		labels=labels.to(device)
		for s in sentence_features:
			batch_to_device(s, device)
		rep_one, rep_two = [self.online_encoder(sentence_feature) for sentence_feature in sentence_features]
		online_pred_one, online_pred_two = rep_one['sentence_embedding'], rep_two['sentence_embedding']
		online_pred_one, online_pred_two = self.online_predictor_1(online_pred_one), self.online_predictor_1(online_pred_two)
		online_pred_one, online_pred_two = self.online_predictor_2(online_pred_one), self.online_predictor_2(online_pred_two)
		online_pred_one, online_pred_two = self.online_predictor_3(online_pred_one), self.online_predictor_3(online_pred_two)
		for s in target_sentence_features:
			batch_to_device(s, device)
		with torch.no_grad():
			target_one, target_two = [self.target_encoder(sentence_feature) for sentence_feature in target_sentence_features]
			target_proj_one, target_proj_two = target_one['sentence_embedding'],  target_two['sentence_embedding']
			target_proj_one, target_proj_two = self.target_predictor_1(target_proj_one), self.target_predictor_1(target_proj_two)


		loss_one = loss_fn(online_pred_one, target_proj_two.detach(),labels)
		loss_two = loss_fn(online_pred_two, target_proj_one.detach(),labels)

		loss = loss_one + loss_two

		return loss.mean()

