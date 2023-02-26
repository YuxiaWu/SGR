import os
import json
data_path = '/storage_fast/yxwu/data'
with open(os.path.join(data_path,'DATA_SPLIT_Feb_5.json'), "r", encoding="utf-8") as f:
    data_split = json.load(f) # data_split: {'train':[],'test':[], 'val':[]}
dial_list = data_split['train'] # the list of the dial index in the mode dataset    
dial_path = os.path.join(data_path,'final_dials_bs_accumulated') # the json file of each dial

num_turn = 0
goal_id_all = []
image_turn_all = []
recommend_turn_all = []
turn_interval_image_recommend = []
next_turn_rec=0
turn_with_image = 0
next_turn_rec2 = 0
image_turn_is_0 = 0
image_turn_is_1 = 0

for json_file_name in dial_list:
    dial_idx = json_file_name
    with open(os.path.join(dial_path, json_file_name + '.json'),"r", encoding="utf-8") as f:
        data = json.load(f)

    dials = data['dialogue']

    num_turn+=len(dials)
    confirm = True
    recommend_turn = 0
    image_turn = []
    recommend_turn = []
    recommend_turn2 = []
    
    for turn_idx in range(len(dials)-1):
        
        bstate = dials[turn_idx]['bstate']
        turn_label = dials[turn_idx]['user']['turn_label']
        image=False
        for slot_value, action in turn_label.items():
            if isinstance(action, list): #['recommend']
                action = action[0] 
            if action=='inform':
                slot_value = slot_value.split(':')
                slot_value = [i.lower() for i in slot_value]
                slot = slot_value[0]
                #value = slot_value[1]
                if slot=='img_gts' and len(slot_value[1])>0:
                    image_turn.append(turn_idx)
                    image=True
                    turn_with_image+=1
                    
        
        agent_action = dials[turn_idx+1]['agent']['dialog_act']
        for dial_slot_value, action_i in agent_action.items():  
            if isinstance(action_i, list): #['recommend']
                action_i = action_i[0]          
            if action_i=='recommend':
                recommend_turn.append(turn_idx+1)
                if image:
                    print('next_turn_rec_dial_idx: ', dial_idx)
                    #next_turn_rec_dial.append(dial_idx)
                    next_turn_rec+=1
                    if turn_idx==0:
                        image_turn_is_0+=1
                    if turn_idx==1:
                        image_turn_is_1+=1                    
        
        try:
            agent_action = dials[turn_idx+2]['agent']['dialog_act']
            for dial_slot_value, action_i in agent_action.items():  
                if isinstance(action_i, list): #['recommend']
                    action_i = action_i[0]          
                if action_i=='recommend':
                    recommend_turn2.append(turn_idx+2)
                    if image:
                        print('2next_turn_rec_dial_idx: ', dial_idx)
                        next_turn_rec2+=1
                        if turn_idx==0:
                            image_turn_is_0+=1
                        if turn_idx==1:
                            image_turn_is_1+=1


        except:
            pass
    

    if len(image_turn)>0:
        print('image_turn: ',image_turn)
        print('recommend_turn: ',recommend_turn)
        print('dial_idx: ', dial_idx)
        print('next_turn_rec: ', next_turn_rec)
        print('turn_with_image: ', turn_with_image)
        print('next_turn_rec2: ', next_turn_rec2)
        print('image_turn_is_0: ', image_turn_is_0)
        print('image_turn_is_1: ', image_turn_is_1)

        print('--------------')

