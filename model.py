import random
import time
import numpy as np
import torch
import torch.nn as nn
from Evaluation import Evaluation
from sgcn_modify import SignedGraphConvolutionalNetwork
from tqdm import trange
import copy

class SGR(nn.Module):
	def __init__(self, args, device, user_size, Data):
		super(SGR, self).__init__()
		
		self.args = args
		self.device = device
		self.Data = Data
		self.UserEmb = nn.Embedding(user_size, args.hidden_dim)  # it is for user embedding init
		self.node_count = self.Data.node_count  
		self.ItemEmb = nn.Embedding(self.node_count-1, args.hidden_dim)
		
		self.user_id_in_graph = self.Data.user_id_in_graph  # add the user node in the graph to construct the state graph
		self.node_feature_kg = self.Data.node_feature_kg.to(self.device) 

		
		if self.args.pre_train:
			self.ItemEmb.weight.data.copy_(self.Data.node_feature_kg)

		if self.args.mlp:
			self.fc1 = nn.Linear(self.args.hidden_dim*2, self.args.fc2_in_dim ) 
			self.fc2 = nn.Linear(self.args.fc2_in_dim, self.args.fc2_out_dim)
			self.linear_out = nn.Linear(self.args.fc2_out_dim, 1)
			self.mlp = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.linear_out)		   

		self.SGCNmodel = SignedGraphConvolutionalNetwork(
			self.device, self.args, self.node_count).to(self.device)

		self.loss_function = nn.BCEWithLogitsLoss()
		self.cal_metrics = Evaluation()
		
	def train_one_sample(self, bstate_pos, bstate_neg, agent, user_id, act_pre):
		# 1. get user feature based on user id ......................................
		#user_id = torch.LongTensor(user_id).to(self.device)
		#user_feature = self.UserEmb(user_id) #1*hidden_dim 
		
		# 2. construct the state graph based on the bstate ...........................
		positive_edges, negative_edges, y, X = self.graph_update(user_id, bstate_pos, bstate_neg)
		
		# 3. get the node_features by SGCN ..........................................
		loss_gcn, node_features = self.SGCNmodel(
			positive_edges, negative_edges, y, X
			)

		del positive_edges, negative_edges, y, X
		del user_id, bstate_pos, bstate_neg

		# 4. compute the score loss .................................................
		gt_action = sorted(list(agent.keys()))
		pre_action = [i[0] for i in act_pre]  # act_pre: [('inform',), ('recommend',)]  
		
		if pre_action==gt_action:				 
			slot_score, value_score, rec_score = self.get_score(node_features)
			del node_features
			if 'request' in set(pre_action):
				loss_slot, _ = self.rank_loss(agent, slot_score, act = 'request')
				loss_rank = loss_slot
				del loss_slot  
			if 'inform' in set(pre_action):
				loss_value, _= self.rank_loss(agent, value_score, act = 'inform')
				try:
					loss_rank = loss_rank + loss_value
				except:
					loss_rank = loss_value
				del loss_value
			if 'recommend' in set(pre_action):
				loss_rec, _ = self.rank_loss(agent, rec_score, act = 'recommend')
				try:
					loss_rank = loss_rank + loss_rec
				except:
					loss_rank = loss_rec
				del loss_rec 
			del slot_score, value_score, rec_score 
			try:  # loss_rank may be []
				loss_gcn = loss_gcn + loss_rank
				del loss_rank
			except:
				pass
		del pre_action, gt_action
		return loss_gcn
			   
	def graph_update(self, user_id, bstate_pos, bstate_neg):
		"""
		update the graph based on the past bstate up to the current turn
		the bstate is obtained based on the utterances
		add user node, and pos/neg edge into the previous graph
		input: 
			bstate history
			previous graph   self.kg_db
		output:
			new graph
		"""
		new_pos, new_neg = [], []
		new_pos.extend(self.Data.edges_kg["positive_edges"])
		new_neg.extend(self.Data.edges_kg["negative_edges"])
		# add user node and edges
		for idx, slot_value_id in enumerate(bstate_pos):
			new_pos.append([self.user_id_in_graph, int(slot_value_id)]) 
		for idx, slot_value_id in enumerate(bstate_neg):
			new_neg.append([self.user_id_in_graph, int(slot_value_id)])
		positive_edges = torch.LongTensor(new_pos).T.to(self.device)  # 2*pos_size  # line 0: from ids， line 2: to ids
		negative_edges = torch.LongTensor(new_neg).T.to(self.device)  # 2*neg_size
		ecount = len(new_pos) + len(new_neg)
		y = np.array([0]*len(new_pos) + [1]*len(new_neg) +[2]*(ecount*2))
		
		y = torch.LongTensor(y).to(self.device)
		# wish to classify whether a pair of node embeddings are from users with a positive, negative, or no link between them
		# size y = (pos_size+neg_size)*3 
		user_id = torch.LongTensor(user_id).to(self.device)
		user_feature = self.UserEmb(user_id) #1*hidden_dim
		node_feature = self.ItemEmb.weight
		X = torch.vstack((node_feature, user_feature))
		del user_id, idx, slot_value_id, ecount,new_pos, new_neg
		return positive_edges, negative_edges, y, X

	def inference_one_sample(self, model, bstate_pos, bstate_neg, act_pre, agent, user_id, batch_gt_venue, request_hist, item_rec_for_IMR, epoch_train ):
		positive_edges, negative_edges, y, X = self.graph_update(user_id, bstate_pos, bstate_neg)
		model_copy = copy.deepcopy(model)
		test_model = model_copy.SGCNmodel
		test_model.train()

		optimizer = torch.optim.Adam(test_model.parameters(),
											lr=self.args.learning_rate,
										weight_decay=self.args.weight_decay)
		
		
		gt_action = sorted(list(agent.keys()))
		pre_action =  [i[0] for i in act_pre]
		user_id = str(user_id.numpy()[0])
		
		metrics_EMR, metrics_IMR = [], []		
		inform_EMR, request_EMR, recommend_EMR = [], [], []
		start_time = time.time()
		if 'recommend' in set(agent.keys()):
			gt_item_id = [i[0] for i in agent['recommend']]
			gt_item_id = list(range(len(gt_item_id)))
			item_rec_for_IMR[user_id]['gt_item_id'].extend(gt_item_id)


		if 'recommend' in set(pre_action):
			
			if self.args.test_GCN:
				epochs = trange(int(self.args.test_GCN_epoch), desc="Loss")
				for epoch in epochs:
					optimizer.zero_grad()
					loss_gcn, node_features = test_model(positive_edges, negative_edges, y, X)
					loss_gcn.backward(retain_graph=True)
					epochs.set_description("SGCN (Loss=%g)" % round(loss_gcn.item(), 4))
					optimizer.step()
			else:
				with torch.no_grad():
					loss_gcn, node_features = test_model(positive_edges, negative_edges, y, X)
			with torch.no_grad():
				slot_score, value_score, rec_score = self.get_score(node_features)
			
			

			if len(batch_gt_venue)>0:
				item_for_sort_all = []
				item_for_sort_all = list(range(self.args.len_items))
				item_for_sort = []
				gt_venue_id = [int(i) for i in batch_gt_venue]
				random.shuffle(item_for_sort_all)
				random.seed(self.args.seed)
				while True:
					neg = random.sample(item_for_sort_all, 1)
					if neg[0] not in gt_venue_id:
						item_for_sort.append(neg[0])
					if len(item_for_sort)==10-len(gt_venue_id):
						break
				item_for_sort = gt_venue_id + item_for_sort
				
				gt_score = [1]*len(gt_venue_id) + [0]*(10-len(gt_venue_id))
				gt_score = torch.Tensor(gt_score).cuda()
				rec_score2 = rec_score[item_for_sort]
				_, sort_rec = torch.sort(rec_score2, descending=True)  # len 10 
				item_rec_for_IMR[user_id]['pre_item_id'].extend([sort_rec.tolist()[0]])
						
		if pre_action==gt_action:

			if 'recommend' in pre_action:
				loss_rec = self.loss_function(rec_score2, gt_score)
				
				try:
					loss_rank = loss_rank + loss_rec
				except:
					loss_rank = loss_rec
				
				item_for_sort = []
				gt_venue_id_turn = [int(agent['recommend'][0])]
				random.seed(self.args.seed)
				
				while True:
					neg = random.sample(item_for_sort_all, 1)
					if neg[0] not in gt_venue_id_turn:
						item_for_sort.append(neg[0])
					if len(item_for_sort)==9:
						break
				random.seed(self.args.seed)
				item_for_sort = gt_venue_id_turn + item_for_sort
				
				gt_score = [1] + [0]*9
				gt_score = torch.Tensor(gt_score).cuda()
				rec_score2 = rec_score[item_for_sort]
				_, sort_rec = torch.sort(rec_score2, descending=True)  # len 10 
				
				gt_item_id = torch.LongTensor([0]).cuda()
				print('gt_item_id: ', gt_item_id)
				metrics_rec = self.get_metrics(sort_rec, gt_item_id)
				print('metrics_rec_act: ', metrics_rec)
		
				metrics_EMR.append(metrics_rec)			
				recommend_EMR.append(metrics_rec)

			else:
				
				if self.args.test_GCN:

					epochs = trange(int(self.args.test_GCN_epoch), desc="Loss")	
					for epoch in epochs:
						optimizer.zero_grad()
						loss_gcn, node_features = test_model(positive_edges, negative_edges, y, X)
						loss_gcn.backward(retain_graph=True)
						epochs.set_description("SGCN (Loss=%g)" % round(loss_gcn.item(), 4))
						optimizer.step()
				else:
					with torch.no_grad():
						loss_gcn, node_features = test_model(positive_edges, negative_edges, y, X)
					
				with torch.no_grad():
					slot_score, value_score, rec_score = self.get_score(node_features)
				
				
				if 'request' in pre_action:
					loss_slot, gt_id = self.rank_loss(agent, slot_score, act = 'request')
					try:
						loss_rank = loss_rank + loss_slot
					except:
						loss_rank = loss_slot

					_, sort_slot = torch.sort(slot_score, descending=True)
					if len(gt_id)>0:
						metrics_slot = self.get_metrics(sort_slot, gt_id)
						metrics_EMR.append(metrics_slot)
						request_EMR.append(metrics_slot)

				if 'inform' in pre_action:
					
					loss_value, gt_id = self.rank_loss(agent, value_score, act = 'inform')
					try:
						loss_rank = loss_rank + loss_value
					except:
						loss_rank = loss_value
					value_for_sort_all = list(range(1768, 6056, 1))
					value_for_sort = []
					bstate_pos = [idx[0].numpy().tolist() for idx in bstate_pos]
					for v in value_for_sort_all:
						if v not in bstate_pos:
							value_for_sort.append(int(v-1768))
					value_score = value_score[value_for_sort]
					_, sort_value = torch.sort(value_score, descending=True)
					sort_value = torch.Tensor(value_for_sort)[sort_value.tolist()].cuda() 
					
					if len(gt_id)>0:
						metrics_value = self.get_metrics(sort_value,gt_id)
						metrics_EMR.append(metrics_value)
						inform_EMR.append(metrics_value)
	   
			loss = loss_gcn.item() + loss_rank.item()
		else:
			try:
				loss = loss_gcn.item()
			except:
				loss=0

		return loss, metrics_EMR, item_rec_for_IMR, [inform_EMR, request_EMR, recommend_EMR]
	

	def rank_loss(self, agent, slot_score, act = 'request'):
		gt_slot = agent[act]
		gt_slot_id = torch.LongTensor(gt_slot)

		if act=='recommend':
			gt_slot_id = gt_slot_id[gt_slot_id<1768]						 
		else:				
			if act =='inform':
				gt_slot_id = gt_slot_id-1768
			gt_slot_id = torch.LongTensor(gt_slot_id)
			gt_slot_id = gt_slot_id[gt_slot_id>=0]

		gt_slot_id = gt_slot_id.to(self.device)
		# inform venuename			
		if len(gt_slot_id)>0:
			gt_score = torch.zeros(1, slot_score.size(0)).to(self.device).scatter_(1, gt_slot_id.unsqueeze(0),1)
			
			loss_rank = self.loss_function(slot_score.unsqueeze(0), gt_score)  # 1*20
		else:
			loss_rank = []
			print('act: ', act)
			print('agent: ', agent)
			print('len gt_slot_id is 0')
		return loss_rank, gt_slot_id
		
	def get_metrics(self, sort_slot, gt_slot_id):
		metrics_slot = []
		for topk in [1, 3, 5]:
			metrics_slot.append(self.cal_metrics.recall_k(sort_slot, gt_slot_id, k=topk))
		return metrics_slot
		
	def get_score(self, features):
		item_feature = features[:-1,:]  # 6056*64
		user_feature = features[-1,:].unsqueeze(0).repeat((item_feature.size(0)),1)  # 6056*64
		if self.args.mlp:
			concat = torch.cat((user_feature, item_feature),1).float() # 6056*128
			value_score_all = self.mlp(concat) # 6056,1
			value_score_all  = value_score_all.squeeze()
			del user_feature, concat		
		else:
			value_score_all =  torch.cosine_similarity(user_feature, item_feature, dim=1) # 6056
		slot_score = [value_score_all[v].mean() for k, v in self.Data.slot_has_values_id.items()]		
		slot_score = torch.stack(slot_score).to(self.device)
		value_score = value_score_all[self.args.len_items:]
		item_score = value_score_all[:self.args.len_items]
		return slot_score, value_score, item_score
		
