import json
import numpy as np
import os
import random
import pandas as pd
import torch
np.random.seed(0)
class DialogRecommendEnv(object):
    def __init__(self, args, device,dataset, act_predict_model, SGR_model, max_turn=15, mode='test', ask_num=1):
        self.mode = mode
        self.args = args
        self.device = device
        self.max_turn = max_turn    #MAX_TURN
        self.dataset = dataset
        self.item_length = 1768
        self.value_feature_length = len(self.dataset.slot_value_map) - self.item_length

        self.kg_db = self.dataset.kg_db  # from:   to:   sign: 

        self.act_predict_model = act_predict_model.to(self.device)
        self.rank_model = SGR_model.to(self.device)
        
        # action parameters
        self.ask_num = ask_num
        self.rec_num = 1

        # user's profile

        self.user_acc_feature = []  # user accepted value_feature which asked by agent
        self.user_rej_feature = []  # user rejected value_feature which asked by agent
        self.cand_items = []   # candidate items
        self.bstate = [] #bstate which records accpted features
        
        self.agent_request_slots = [0, 1,2]  # requested slot by agent

        #user_id  item_id   cur_step   cur_node_set
        self.user_id = None
        self.target_item = None
        self.cur_conver_step = 0        #  the number of conversation in current step

        #init seed & init user_dialog_dict
        if mode == 'valid':
            self.user_dialog_dict = self.__user_dialog_dict_init__() # init self.user_dialog_dict
            self.user_length = len(self.user_dialog_dict)
        elif mode == 'test':
            self.user_dialog_dict = self.__user_dialog_dict_init__() # init self.user_dialog_dict
            self.user_num = len(list(self.user_dialog_dict.keys()))
            self.test_num = 0
        pass
        self.action_space = 3

    def __user_dialog_dict_init__(self):   #Load user dialog dict
        user_dialog_dict = {}
        for json_file_name in self.dataset.dial_list:
            dial_idx = json_file_name
            with open(os.path.join(self.dataset.dial_path, json_file_name + '.json'),"r", encoding="utf-8") as f:
                data = json.load(f)
            user_id = self.dataset.dial2user[dial_idx]
            gt_venue_id = [] # all gt venue of this dialogue
            for goal in data['goal']['content']:
                venuename = goal['venueName']
                venuename = [i.lower() for i in venuename.split(' ')]
                venuename = ' '.join(venuename)
                slot_value = '_'.join(['venuename', venuename])
                slot_value = slot_value.replace('-', ' ')
                venue_id = self.dataset.get_venue_id(slot_value) 
                if venue_id!='none': 
                    gt_venue_id.append(venue_id)
            if len(gt_venue_id)>0:
                user_dialog_dict[user_id] = gt_venue_id
        return user_dialog_dict

    def reset(self, init_first=True):
        self.cur_conver_step = 0   #reset cur_conversation step
        if self.mode == 'valid':
            pass
        elif self.mode == 'test':
            if init_first is True:
                users = list(self.user_dialog_dict.keys())
                self.user_id = users[self.test_num]  #TODO
                self.test_num += 1 #Next user
                self.user_target_items_id =self.user_dialog_dict[self.user_id]
                self.task_num = len(self.user_target_items_id)
                self.dialog_task_ind = 0  # record the sucessfully finish task number
                # == init target_item & user_inform_value
                self.target_item = self.user_target_items_id[self.dialog_task_ind]
                self.user_inform_fea_ids = []
    
                inform_slots = list(range(3,20))
                self.item_dict_target = self.dataset.item_dict[str(self.target_item)]
                for slot, value in self.item_dict_target.items():
                    if slot=='venuename':
                        continue
                    if int(slot) in set(inform_slots):
                        self.agent_request_slots.append(int(slot))
                        self.user_inform_fea_ids.extend(value)
            else:
                # == init target_item & user_inform_value
                self.target_item = self.user_target_items_id[self.dialog_task_ind]
                self.user_inform_fea_ids = []
                inform_slots = list(range(3,20))
                self.item_dict_target = self.dataset.item_dict[str(self.target_item)]
                for slot, value in self.item_dict_target.items():
                    if slot=='venuename':
                        continue                    
                    if int(slot) in set(inform_slots):
                        self.agent_request_slots.append(int(slot))
                        self.user_inform_fea_ids.extend(value)

        # init user's profile
        print('-----------reset state vector------------')
        self.user_acc_feature = []  # user accepted value_feature which asked by agent
        self.user_rej_feature = []  # user rejected value_feature which asked by agent
        self.cand_items = list(range(self.item_length))  # candidate items
        self.bstate = []  # bstate which records accpted features
        self.agent_request_slots = [0,1,2]
        self.bstate_pos = []
        self.bstate_neg = []
        self.bstate_seq = ''
        print('user_id:{}, target_item:{}'.format(self.user_id, self.target_item))
        # init user_greet:  select a favor feature from user_inform_feas
        print('type_target_item: ', type(self.target_item))
        self.user_acc_feature = [np.random.choice(self.user_inform_fea_ids)]
        # init user prefer feature
        self.cur_conver_step += 1
        # init bstate
        self.bstate.extend(self.user_acc_feature.copy())
        self.bstate_pos.extend(self.user_acc_feature.copy())
        value2slot = dict([i,key] for key,val in self.dataset.slot_has_values_id.items() for i in val )
        slot_id = [value2slot[i] for i in self.user_acc_feature]
        slot_value_id_neg = []
        for slot_id_i in slot_id:
            if int(slot_id_i) in self.dataset.slot_id_with_neg:
                neg_values = set(self.dataset.slot_has_values_id[slot_id_i]) - set(self.user_acc_feature)
                slot_value_id_neg.extend(list(neg_values))
            self.bstate_neg.extend(slot_value_id_neg)
        
        value2name = dict([val,key] for key,val in self.dataset.slot_value_map.items() )
        for acc_feature in self.user_acc_feature:
            self.bstate_seq+=' '.join(value2name[acc_feature].split('_'))+' inform;'
        
        self._update_cand_items(acc_feature=self.user_acc_feature, rej_feature=[])
        self.start_cand_item = self.cand_items

    def step(self, action):   #action:0  ask   action:1  request action:2 recommend   setp=MAX_TURN  done
        done = 0
        self.user_acc_feature = list(set(self.user_acc_feature))
        print('---------------step:{}-------------'.format(self.cur_conver_step))
        if self.cur_conver_step >= self.max_turn:
            print('--> Maximum number of turns reached !')
            recommend_success = False
            done = 1
        else:
            recommend_success = False

            if action == 'inform':
                done = 0
            else:
                positive_edges, negative_edges, y, X = self.rank_model.graph_update([self.user_id], self.bstate_pos, self.bstate_neg)
                        
                _, node_features = self.rank_model.SGCNmodel(positive_edges, negative_edges, y, X)
                slot_score, value_score, rec_score= self.rank_model.get_score(node_features)
                rec_score_list = rec_score.tolist()
                now_cand_item = self.cand_items_with_attrs

                now_cand_item_score = [rec_score_list[item_id] for item_id in now_cand_item]
                now_item_score_tuple = list(zip(now_cand_item, now_cand_item_score))
                now_item_score_tuple = sorted(now_item_score_tuple, key=lambda x:(x[1],x[0]), reverse=True)
                target_item_score = rec_score_list[self.target_item] 
                print('target_item: ', self.target_item)
                print('target_item_score: ', target_item_score)
                
                if len(self.cand_items)<30:
                    print('now item_score_tuple: ', now_item_score_tuple)

                if action == 'request': # request a slot
                    print('-->action: request a slot values')
                    select_slot_id = self._get_slot_id(slot_score)  # get max_entropy slot id
                    print(f'==get slot value id {select_slot_id}')
                    print('agent_request_slots: ', self.agent_request_slots)
                    
                    if select_slot_id!=8:
                        self.agent_request_slots.append(select_slot_id)
                    acc_feature, rej_slot = self._request_answer(slot_id=select_slot_id, ans_num=1)
                    self.user_acc_feature += acc_feature
                    print(f'-->user answer request features: {acc_feature}')
                    self._update_cand_items(acc_feature=self.user_acc_feature, rej_feature=[])  # update cand_items
                    
                    # ==update bstate
                    self.user_acc_feature = list(set(self.user_acc_feature))
                    self.bstate_pos.extend(self.user_acc_feature.copy())
                    value2slot = dict([i,key] for key,val in self.dataset.slot_has_values_id.items() for i in val )
                    slot_id = [value2slot[i] for i in self.user_acc_feature]
                    slot_value_id_neg = []
                    for slot_id_i in slot_id:
                        if int(slot_id_i) in self.dataset.slot_id_with_neg:
                            neg_values = set(self.dataset.slot_has_values_id[slot_id_i]) - set(self.user_acc_feature)
                            slot_value_id_neg.extend(list(neg_values))
                        self.bstate_neg.extend(slot_value_id_neg)
                    
                    value2name = dict([val,key] for key,val in self.dataset.slot_value_map.items() )
                    for acc_feature in self.user_acc_feature:
                        seq = ' '.join(value2name[acc_feature].split('_')) + ' inform;'
                        if seq not in self.bstate_seq:
                            self.bstate_seq+=seq

                if action == 'recommend':  #recommend items
                    print('-->action: recommend items')
                    #select topk candidate items to recommend
                    cand_item_score = self._item_score(rec_score)  # bilstm model
                    item_score_tuple = list(zip(self.cand_items, cand_item_score))
                    sort_tuple = sorted(item_score_tuple, key=lambda x: x[1], reverse=True)
                    self.cand_items, cand_item_score = zip(*sort_tuple)
                    #===================== rec update=========
                    recom_items = self.cand_items[: self.rec_num]  # TOP k item to recommend
                    print('cand_items: ', self.cand_items)
                    print('recom_items: ', recom_items)
                    print('target_item: ', self.target_item)
                    
                    if self.target_item in recom_items:
                        print('-->Recommend successfully!')
                        self.dialog_task_ind += 1  # Next Task
                        done = 1
                        print(f'user-{self.user_id} finish all [{self.dialog_task_ind}/{self.task_num} tasks!]')
                        recommend_success = True
                    else:
                        self.bstate_neg.extend(list(recom_items))
                        self.cand_items = self.cand_items[self.rec_num:]  # update candidate items
                        print('-->Recommend fail !')

            print(f'-->user acc  all features: {self.user_acc_feature}')
            print(f'-->cand item_length: {len(self.cand_items)}')
        return done, recommend_success


    def _get_user_inform(self, number=1, prob=0.8):  # get user inform_fea_id
        """
        :param number:  the number of user inform's features
        :return: inform slot-value feature_id
        """
        cand_inform_num = len(self.user_inform_fea_ids)
        self.user_inform_fea_ids = list(set(self.user_inform_fea_ids) - set(self.user_acc_feature))
        if cand_inform_num == 0:  #check
            return []
        inform_feature = self.user_inform_fea_ids[:number]
        self.user_acc_feature += inform_feature  # update user acc_fea
        self.user_inform_fea_ids = self.user_inform_fea_ids[number:]  # update self.inform_fea_ids
        return inform_feature

    def _item_score(self, value_score):  
        value_score = value_score.tolist()
        cand_item_score = []
        for item_id in self.cand_items:
            pred_score = value_score[item_id]
            cand_item_score.append(pred_score)
        return cand_item_score

    def _ask_feature(self, value_score):
        slot_score = np.array(value_score.cpu().numpy())# max_entropy slot id
        slot_score = (slot_score-np.min(slot_score))/(np.max(slot_score)-np.min(slot_score))
        sort_ind = slot_score.argsort()[::-1].tolist()
        print('value_sort_ind: ', sort_ind)
        select_slot_id = sort_ind[0]
        return select_slot_id
       
    def _get_slot_id(self,slot_score):  
        slot_score = np.array(slot_score.cpu().numpy())# max_entropy slot id
        slot_score = (slot_score-np.min(slot_score))/(np.max(slot_score)-np.min(slot_score))
        sort_ind = slot_score.argsort()[::-1].tolist()
        print('slot_sort_ind: ', sort_ind)

        for idx in set(self.agent_request_slots):
            if idx in sort_ind:
                sort_ind.remove(idx)
        print('agent_request_slots: ', self.agent_request_slots)
        print('sort-slot-remove: ', sort_ind)
        
        select_slot_id = sort_ind[0]

        return select_slot_id


    def _update_cand_items(self, acc_feature, rej_feature):  
        if len(acc_feature):  # accept feature
            print('=== ask acc: update cand_items')
            print('acc feature: ', acc_feature)
            for feature_id in acc_feature:
                feature_items = self.dataset.slot_value2item[str(feature_id)]
                self.cand_items = set(self.cand_items) & set(feature_items)  
            self.cand_items = list(self.cand_items)
            self.cand_items_with_attrs = list(self.cand_items)

    def _request_answer(self, slot_id, ans_num=1):
        print('keys: ',self.item_dict_target.keys())
        try:
            acc_feature = [np.random.choice(self.item_dict_target[str(slot_id)])]
            rej_slot = []
        except:
            acc_feature = []
            rej_slot = [slot_id]
        return acc_feature, rej_slot
