# the dataset 
import torch
from torch.utils.data import Dataset
import os
import json
import nltk.stem as ns
import pickle
import pandas as pd
from collections import OrderedDict 
import numpy as np
from collections import Counter
import random
class MmdialData(Dataset):
    def __init__(self, args, mode = 'train'):
        super(MmdialData, self).__init__()
        print('------------------------------------------')
        print('mode: ', mode)
        self.errors = 0
        self.args = args
        self.mode = mode        
        with open(os.path.join(self.args.data_path,'DATA_SPLIT_Feb_5.json'), "r", encoding="utf-8") as f:
            self.data_split = json.load(f) # data_split: {'train':[],'test':[], 'val':[]}
        self.dial_list = self.data_split[self.mode] # the list of the dial index in the mode dataset    
        self.dial_path = os.path.join(self.args.data_path,'final_dials_bs_accumulated') # the json file of each dial

        self.dial2user = self.get_dial2user()
        if mode=='train':
            print('user_size', len(self.dial2user))
                  
        with open(os.path.join(self.args.data_path,'image_map.json'), "r", encoding="utf-8") as f:
            self.image_map = json.load(f)

        image_sim_file = 'image_user_sim_db_id.json'

        with open(os.path.join(self.args.data_path, image_sim_file), "r", encoding="utf-8") as f:
            self.image_sim = json.load(f) 
        
        with open(os.path.join(self.args.data_path,'dial_turn_id_with_image.json'), "r", encoding="utf-8") as f:
            self.dial_turn_id_with_image = json.load(f)  

        self.slot_with_neg = set(['venueScore', 'Credit Cards', 'Menus', 'venueNeigh', 'Parking', \
                     'Dining Options', 'Restroom', 'Wi-Fi', 'Music', 'Outdoor Seating', \
                     'Wheelchair Accessible', 'Smoking', 'Reservations', 'Drinks', 'price'])
        # venue name, telephone, image. open span
        with open(os.path.join(self.args.data_path,'slot_value_map.json'), "r", encoding="utf-8") as f:
            self.slot_value_map = json.load(f) # "venuename_sentosa beach": 0,
        with open(os.path.join(self.args.data_path,'slot_id_map.json'), "r", encoding="utf-8") as f:
            self.slot2id = json.load(f)  # "venuename": 0,       
        with open(os.path.join(self.args.data_path, 'slot_values_id.json'), "r", encoding="utf-8") as f:
            self.slot_has_values_id = json.load(f) # the values id belonging to the slot "0": [0, 1, 2, 3, 4, 5,6,]
        with open(os.path.join(self.args.data_path, 'slot_value2item.json'), "r", encoding="utf-8") as f:
            self.slot_value2item = json.load(f)
        with open(os.path.join(self.args.data_path, 'item_dict.json'), "r", encoding="utf-8") as f:
            self.item_dict = json.load(f)
        self.slot_id_with_neg = set([self.slot2id[i.lower()] for i in self.slot_with_neg])

        # read the KG database,	 the node, edges, features about the pre_training
        self.node_count = len(self.slot_value_map)+1  # with user node  6057
        # construct MMKG
        MMKG_path = './mmdial_csv/'

        if  not os.path.exists(MMKG_path):
            os.makedirs(MMKG_path)
        MMKG_csv = os.path.join(MMKG_path,'mmdial_signed_'+str(self.args.neglink_num) + '.csv')
        print('MMKG_csv: ', MMKG_csv)
        if  not os.path.exists(MMKG_csv):
            print('constructing MMKG.....')
            sign_data = pd.DataFrame(columns=('from','to','sign'))
            for venue_id, slot_value in self.item_dict.items():
                venue_id = int(venue_id)
                for slot_id, slot_value_ids in slot_value.items():
                    if slot_id=='venuename':
                        continue

                    if slot_value_ids==[venue_id]:
                        continue

                    for slot_value_id in slot_value_ids:
                        sign_data = sign_data.append(pd.DataFrame({'from':[venue_id],'to':[slot_value_id],'sign':[1]}), ignore_index=True)
                    if int(slot_id) in self.slot_id_with_neg:
                        neg_values = set(self.slot_has_values_id[slot_id])-set(slot_value_ids)
                        if self.args.neglink_num<1:
                            neg_values_selected = random.sample(list(neg_values), round(self.args.neglink_num*len(neg_values)))
                        else:
                            neg_values_selected = neg_values
                        for value_i in neg_values_selected:
                            sign_data = sign_data.append(pd.DataFrame({'from':[venue_id],'to':[value_i],'sign':[-1]}), ignore_index=True)
            sign_data.to_csv(MMKG_csv, index=0)
        
        self.kg_db = pd.read_csv(MMKG_csv).values.tolist()
        
        edges = {}
        edges["positive_edges"] = [edge[0:2] for edge in self.kg_db if edge[2] == 1]
        edges["negative_edges"] = [edge[0:2] for edge in self.kg_db if edge[2] == -1]
        edges["ecount"] = len(self.kg_db)
        edges["ncount"] = self.node_count-1  # the node_count in KG
        self.edges_kg = edges
        feature = np.array(pd.read_csv(self.args.pretrain_features_path))  # the pre_trained node features in KG
        self.node_feature_kg = torch.from_numpy(feature[:,1:]) # the first column is the node id  0:xxxxx node_id: feature
        self.user_id_in_graph = int(self.node_count-1)  # add the user node in the graph to construct the state graph
        
        act_prediction_path = os.path.join(self.args.data_path, 'act_prediction_result',  mode + '.json')
        with open(act_prediction_path, 'r', encoding="utf-8") as f:
            self.act_prediction_result = json.load(f) 
        self.act_prediction_result = iter(self.act_prediction_result)

        if self.args.with_image:
            sample_path = os.path.join(self.args.data_path, 'sample_neglink_' + str(self.args.neglink_num), 'with_image')
        else:
            sample_path = os.path.join(self.args.data_path,'sample_neglink_'+ str(self.args.neglink_num), 'wo_image')

        print('sample_path: ', sample_path)
    
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        print('sample_path: ', sample_path)
        
        self.correct_slot = {
            'open psan': 'open span',
            'opne span': 'open span',
            'open open': 'open span',
            'opan span': 'open span',
            'opens span': 'open span',
            'open spicy': 'open span',
            'oprn span': 'open span',
            'open spam': 'open span',
            'oepn span': 'open span',
            'venuenaddress': 'venueaddress',
            'venueneight': 'venueneigh',
            'menu': 'menus',
            'anything else': 'open span',
        }
        
        if not os.path.exists(os.path.join(sample_path,mode + '_sample_all.json')):
            self.sample_all, self.goal_id, self.prob_request = self.get_sample_all() 
            print('construct the sample.....')
            with open(os.path.join(sample_path, mode + '_sample_all.json'),'w', encoding="utf-8") as f:
                json.dump(self.sample_all, f)
            with open(os.path.join(sample_path , mode + '_goal_id.json'),'w', encoding="utf-8") as f:
                json.dump(self.goal_id, f)                
            with open(os.path.join(sample_path,  mode + '_prob_request.pkl'),'wb') as f:
                pickle.dump(self.prob_request, f)         
        else:
        
            print('load the sample_all.....')
            with open(os.path.join(sample_path,mode + '_sample_all.json'),'r', encoding="utf-8") as f:
                self.sample_all = json.load(f) 
            with open(os.path.join(sample_path, mode + '_goal_id.json'),'r', encoding="utf-8") as f:
                self.goal_id = json.load(f)                 
            with open(os.path.join(sample_path, mode + '_prob_request.pkl'),'rb') as f:
                self.prob_request = pickle.load(f) 
        print('len_sample: ', len(self.sample_all))
        
    
    def get_dial2user(self):
        dial2user = OrderedDict()     
        dial_name = os.listdir(self.dial_path)
        idx = -1
        for name in dial_name:
            dial_idx = name.split('.')[0]
            if dial_idx not in dial2user.keys():
                idx+=1
                dial2user[dial_idx] = int(idx)
        return dial2user

    def get_venue_id(self, slot_value):
        if slot_value=='venuename_art science museum':
           slot_value = 'venuename_artscience museum' 
        try:    
            venue_id = self.slot_value_map[slot_value]
        except:
            try:
                venue_id = self.slot_value_map[slot_value+' ']
            except:
                try:
                    venue_id = self.slot_value_map[slot_value+' ()']
                except:
                    try:
                        venue_id = self.slot_value_map['venuename_' + ' '+venuename]
                    except:
                        try:
                            venue_id = self.slot_value_map[slot_value+' | ']
                        except:
                            try:
                                venue_id = self.slot_value_map['venuename_' + 'the ' + slot_value.split('_')[1]]
                            except:
                                try:
                                    venue_id = self.slot_value_map[slot_value+' (light & water show)']
                                except:
                                    venue_id = 'none'
        return venue_id       

    def get_sample_all(self):
        sample_all = []
        num_turn = 0
        
        goal_id_all = []
        inform_id_all, request_id_all = [], []
        image_turn_all = []
        turn_interval_image_recommend = []
        for json_file_name in self.dial_list:
            dial_idx = json_file_name
            with open(os.path.join(self.dial_path, json_file_name + '.json'),"r", encoding="utf-8") as f:
                data = json.load(f)

            dials = data['dialogue']
            gt_venue_id = [] # all gt venue of this dialogue
            goal_id_all.append(data['goal']['id'])
            for goal in data['goal']['content']:
                venuename = goal['venueName']
                venuename = [i.lower() for i in venuename.split(' ')]
                venuename = ' '.join(venuename)
                slot_value = '_'.join(['venuename', venuename])
                slot_value = slot_value.replace('-', ' ')
                venue_id = self.get_venue_id(slot_value) 
                if venue_id!='none': 
                    gt_venue_id.append(venue_id)
        
            num_turn+=len(dials)
            request_hist = []
            confirm = True
            recommend_turn = 0
            image_turn = False
            for turn_idx in range(len(dials)-1):
                if turn_idx==0 or confirm:
                    request_hist = []
                else:
                    try:
                        request_hist.extend(action_id['request']) # the action of the lasst turn
                    except:
                        pass
                bstate = dials[turn_idx]['bstate']
                agent_action = dials[turn_idx+1]['agent']['dialog_act']
                slot_value_id_pos, slot_value_id_neg, confirm, image_turn = self.get_bstate_id(bstate, dial_idx, turn_idx)
                action_id = self.get_action_id(agent_action, dial_idx, turn_idx+1)
                
                
                if 'recommend' in action_id:
                    recommend_turn = turn_idx+1

                if image_turn:
                    image_turn_all.append(image_turn)
                    if 'recommend' in action_id:
                        turn_interval_image_recommend.append(int(recommend_turn-image_turn))

                act_predict = next(self.act_prediction_result)
                try:
                    request_id_all.extend(action_id['request'])
                except:
                    pass
                try:
                    inform_id_all.extend(action_id['inform'])
                except:
                    pass

                if len(slot_value_id_pos)>0 and len(action_id)>0 and len(act_predict)>0 and len(gt_venue_id)>0:
                    sample_all.append(
                        [
                            slot_value_id_pos, 
                            slot_value_id_neg, 
                            action_id, 
                            self.dial2user[dial_idx], 
                            gt_venue_id, 
                            act_predict,
                            list(set(request_hist))
                        ]
                            )
        
        print('num_turn: ', num_turn)
        if self.mode=='train':
            count_request = dict(Counter(request_id_all)) 
            for i in range(20):
                if not i in count_request.keys():
                    count_request[i] = 0
            count_df = pd.DataFrame.from_dict(count_request, orient='index')
            count_df.sort_index(inplace=True)
            count_df = (count_df-count_df.min())/(count_df.max()-count_df.min())
            prob_request = count_df[0].values
        else:
            prob_request=[]
        print('ok')
        print(set(turn_interval_image_recommend))
        print(len(image_turn_all))
        return sample_all, goal_id_all, prob_request
    
    def __getitem__(self, index):
        slot_value_id_pos, slot_value_id_neg, action_id, user_id, gt_item_id, act_predict, request_hist = self.sample_all[index] 
        return slot_value_id_pos, slot_value_id_neg, action_id, user_id, gt_item_id, act_predict, request_hist
        
    def __len__(self):
        return len(self.sample_all)

    def slot_value_refine(self, slot, value):
        value = value.strip().replace('-', ' ')
        slot_value = [slot, value]
        if slot_value==['delivery', 'no']:
            slot_value = ['dining options', 'no delivery']
        slot, value = slot_value[0], slot_value[1]
        del slot_value
        if slot in self.correct_slot.keys():
            slot = self.correct_slot[slot]
        if slot=='img_gts':
            slot = 'images'
        if 'yes' in value:
            value = 'yes'
        if value=='cocktails':
            value = 'cocktails'
        
        return slot, value

    def get_bstate_id(self, bstate, dial_idx, turn_idx):
        '''
        inputs: 
            bstate: the belief state of current turn
            dial_idx: the dialogue id
            turn_idx: the turn id of this dial
        outputs:  the slot_value_id for construct the state graph
            slot_value_id_pos: the slot_value_id of the positive edges
            slot_value_id_neg: the slot_value_id of the negative edges
        '''
        slot_value_id_pos, slot_value_id_neg = [], []
        for slot_value, action in bstate.items():
            if isinstance(action, list): #['recommend']
                action = action[0] 
            
            confirm = True if 'confirm' in action else False
                     
            if not 'inform' in action: # we don't consider the request state 
                continue 
            slot_value = slot_value.split(':')
            slot_value = [i.lower() for i in slot_value]
            slot = slot_value[0]
            if self.args.with_image: # if we consider images in our dataset
                if slot=='img_gts' and len(self.dial_turn_id_with_image[dial_idx])>0:
                    try:
                        image_name = self.dial_turn_id_with_image[dial_idx][str(turn_idx)]
                    except:
                        turn_id_list = list(self.dial_turn_id_with_image[dial_idx].keys())
                        turn_id_with_image = [i for i in turn_id_list if int(i)<int(turn_idx)]
                        if len(turn_id_with_image)==0:
                            self.errors+=1
                            continue
                        turn_id_with_image = str(turn_id_with_image[-1])
                        image_name = self.dial_turn_id_with_image[dial_idx][turn_id_with_image]
                        del turn_id_list, turn_id_with_image
                    image_turn = turn_idx
                    try:
                        image_id_in_db = str(self.image_sim[image_name])
                        value = image_id_in_db
                        slot_value = [slot, value]
                    except:
                        continue
                elif slot=='img_gts':  # there is no image
                    continue
            else:
                if slot=='img_gts':
                    continue
            
            slot, value = self.slot_value_refine(slot_value[0], slot_value[1])    
            slot_id = self.slot2id[slot]
            slot_value = '_'.join([slot, value])
            try:
                slot_value_id = self.slot_value_map[slot_value]
                slot_value_id_pos.append(slot_value_id)
                if slot_id in self.slot_id_with_neg:
                    neg_values = set(self.slot_has_values_id[str(slot_id)]) - set([slot_value_id])
                    
                    if self.args.neglink_num<1:
                        neg_values_selected_user = random.sample(list(neg_values), round(self.args.neglink_num*len(neg_values)))
                        for value_i in neg_values_selected_user:                    
                            slot_value_id_neg.extend(list(value_i))
                    else:
                        for value_i in neg_values:                    
                            slot_value_id_neg.extend(list(value_i))                        
                    del slot_id, slot_value, slot, value, neg_values
            except:
                pass
        try:
            a=confirm
        except:
            confirm = False

        try:
            b = image_turn
        except:
            image_turn = False
        return slot_value_id_pos, slot_value_id_neg, confirm, image_turn


    def get_action_id(self, agent, dial_idx, turn_idx):
        
        actions = {} # actions['request']: [id1, id2, ...]
        gt_slot, gt_value = [], []
        
        for dial_slot_value, action_i in agent.items():  
            if 'bye' in action_i:
                continue
            if isinstance(action_i, list): #['recommend']
                action_i = action_i[0]          

            if action_i=='request':
                try:
                    slot_id = self.slot2id[dial_slot_value.lower()]
                except:
                    slot_value = dial_slot_value.split(':')
                    slot_value = [i.lower() for i in slot_value]
                    slot = slot_value[0].strip()
                    if slot in self.correct_slot.keys():
                        slot = self.correct_slot[slot]
                    slot_id = self.slot2id[slot]
                gt_slot.append(slot_id)

                if 'request' not in actions.keys():
                    actions[action_i] = []
                actions[action_i].append(slot_id)
                
            else: # action = 'inform', 'recommend'
                slot_value = dial_slot_value.split(':')
                slot_value = [i.lower() for i in slot_value]
                slot = slot_value[0]

                if len(slot_value)>1:
                    if self.args.with_image:
                        if slot=='img_gts' and len(self.dial_turn_id_with_image[dial_idx])>0:
                            slot = 'images'
                            try:
                                image_name = self.dial_turn_id_with_image[dial_idx][str(turn_idx)]
                            except:
                                turn_id_list = list(self.dial_turn_id_with_image[dial_idx].keys())
                                turn_id_list = [int(i) for i in turn_id_list]
                                turn_id_with_image = [i for i in turn_id_list if i<int(turn_idx)]
                                if len(turn_id_with_image)==0:
                                    continue
                                turn_id_with_image = str(turn_id_with_image[-1])
                                image_name = self.dial_turn_id_with_image[dial_idx][turn_id_with_image]
                            try:
                                image_id_in_db = str(self.image_sim[image_name])
                                value = image_id_in_db
                                slot_value = [slot, value]
                            except:
                                continue
                        elif slot=='img_gts':  # there is no image
                            continue 
                    else:
                        if slot=='img_gts':
                            continue

                    slot, value = self.slot_value_refine(slot_value[0], slot_value[1])
                    slot_value = '_'.join([slot, value])
                    
                    try:
                        slot_value_id = self.slot_value_map[slot_value]
                    except:
                        if slot=='venuename':
                            slot_value_id = self.get_venue_id(slot_value)
                        else:
                            pass
                    try:
                        if slot_value_id!='none':
                            if (action_i=='inform' and slot_value_id>1767) or (action_i=='recommend' and slot_value_id<1768):
                                gt_value.append(slot_value_id)
                                if action_i not in actions.keys():
                                    actions[action_i] = []
                                actions[action_i].append(slot_value_id)
                    except:
                        pass
        return actions
