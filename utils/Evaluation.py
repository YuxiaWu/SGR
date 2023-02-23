#coding: utf-8
# author: lu yf
# create date: 2019-11-27 13:14
import math
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, auc
import torch

class Evaluation:
	def map_k(self,sort, y, k):
		sum_precs = 0
		ranked_list = sort[:k]
		hists = 0
		for n in range(len(ranked_list)):
			if ranked_list[n] in y:
				hists += 1
				sum_precs += hists / (n + 1)
		return sum_precs

	def recall_k(self,sort, y, k):
		recall_correct = 0
		for y_i in y:
			if y_i in sort[:k]:
				recall_correct += 1
		return recall_correct/len(y)

	def precision_k(self, sort, y, k):
		precision_correct = 0
		for y_i in y:
			if y_i in sort[:k]:
				precision_correct += 1
		return precision_correct/k

	def ndcg_k(self,sorted_indices, ground_truth, k):
		dcg, pdcg = 0,0
		for i, item in enumerate(sorted_indices[:k]):
			if item in ground_truth:
				dcg += 1 / math.log(i + 2)
		for i in range(min(len(ground_truth), k)):
			pdcg += 1 / math.log(i + 2)
		return dcg / pdcg

	def cal_auc(self, query_set_y_pred, y_label):
		fpr, tpr, thresholds = roc_curve( y_label.squeeze().cpu().data.numpy(), \
												query_set_y_pred.squeeze().cpu().data.numpy(), pos_label=1)
		auc_val = auc(fpr, tpr)
		return auc_val



