"""
online_evaluate_task:

when there is more than one task, 
the input of the gpt will contain the bstate of the former ones

"""



import time
from itertools import count
import torch
from env_agent import DialogRecommendEnv
import sys
sys.path.append('../code/')
#sys.path.append('../code/utils/')
sys.path.append('../code/dialogpt/')
sys.path.append('../code/dialogpt/utils/')
sys.path.append('./utils/')

from param_parser import parameter_parser

from model_rg import DialoGPT
from dataloader import TextDataset

import torch
import random
from MmdialDataPre import MmdialData
from model import SGR
import numpy as np
from transformers import AutoTokenizer
import os
from model_utils import loop, get_device_ids, CustomPaddingTensorCollator, set_max_len
from torch.utils.data import DataLoader

def evaluate(args, device, dataset, act_predict_model, SGR_model,tokenizer, pad_value):
    
    run_name = "online_test_task_1123"
    log = open("log/log_" + run_name + ".txt", "a")
    sys.stdout = log
    
    max_turn = 15
    test_env = DialogRecommendEnv(args, device, dataset, act_predict_model, SGR_model, max_turn=15, mode='test', ask_num=1)  # init env!
    
    tt = time.time()
    start = tt
    SR5, SR10, SR15, AvgT = 0, 0, 0, 0
    SR_turn_15 = [0]* max_turn
    turn_result = []
    result = []
    task_num = 0
    user_size = test_env.user_num
    print('User size in UI_test: ', user_size)

    key2pad_id = {
        'input_ids': pad_value,
        'labels': pad_value
    }

    first_eval = {
        'input_ids': True,
        'attention_mask': True
    }

    for user_idx in list(range(user_size)):  #user_size
        log.flush()
        print('\n================test tuple:{}===================='.format(user_idx))
        print('\n ----user-----')
        print('{} /{}'.format(user_idx, user_size))

        print('task_num of this user: ', task_num )
        test_env.reset()  # Reset environment and record the starting state
        for task_idx in list(range(test_env.task_num)):
            print('\n ----task of this user-----')
            print('{} /{}'.format(task_idx, test_env.task_num))
            task_num += 1
            for t in count():  # user  dialog
                with torch.no_grad():   
                    bstate_seq = test_env.bstate_seq
                    print('bstate_seq: ', bstate_seq)
                    data_file = 'test_online_task_1123.action_prediction'
                    with open(data_file,'w') as f:
                        f.write('<|belief|> ' + bstate_seq + ' <|endofbelief|> <|action|>  <|endofaction|>' + '\n')
                    test_sample = TextDataset.create_data(data_file, tokenizer_or_transformer_model=tokenizer, end_token=' <|endofaction|>', remove_tokens={}, split_token='<|endofbelief|>', split=(1,), training=False, shuffle=False) 
                    collator_eval = CustomPaddingTensorCollator(key2pad_id=key2pad_id, first=first_eval)
                    dataloader_test_act = DataLoader(test_sample, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, collate_fn=collator_eval)
                    try:
                        for bstate_seq_i in zip(dataloader_test_act): #only one sample
                            act_inputs, extras = act_predict_model.prepare(bstate_seq_i[0])
                            outputs = act_predict_model(**act_inputs)
                        
                            results = act_predict_model.make_results(outputs, extras)
                    except:
                        done=1
                        break
                    for _, act_result in results.items(): # only one
                        act_result = act_result[0]
                        pre = act_result['predictions']['rg']
                        pre_token = tokenizer.convert_ids_to_tokens(pre)
                        pre_action = [i[1:] for i in pre_token if i[1:] in set(['inform', 'request', 'recommend'])]
                        pre_action = sorted(pre_action) 
                    print('pre_action: ', pre_action)      
                    for action in pre_action:
                        #if action == 'inform':
                        #    done = 0
                        #else:
                        done, recommend_success = test_env.step(action)
                        print('done: ', done)
                        #print('dialog_task_ind:  ', test_env.dialog_task_ind)
                        if done:
                            if recommend_success:  # recommend successfully
                                SR_turn_15 = [v+1 if i>t  else v for i, v in enumerate(SR_turn_15) ]
                                if t < 5:
                                    SR5 += 1
                                    SR10 += 1
                                    SR15 += 1
                                elif t < 10:
                                    SR10 += 1
                                    SR15 += 1
                                else:
                                    SR15 += 1
                            AvgT += t+1

                            SR = [SR5, SR10, SR15, AvgT]
                            print('SR-until now: ', SR)  
                            print('SR/task_num: ', [SR5/task_num, SR10/task_num, SR15/task_num, AvgT/task_num])
                            result.append(SR) 
                            break #  break action
                    
                    if done==0:
                        test_env.cur_conver_step+=1
                    else:
                        break # break the current task

            # break step, start next task
            if done ==1 and test_env.dialog_task_ind<test_env.task_num:
                test_env.target_item = test_env.user_target_items_id[test_env.dialog_task_ind]
                inform_slots = list(range(3,20))
                test_env.item_dict_target = test_env.dataset.item_dict[str(test_env.target_item)]
                test_env.user_inform_fea_ids = []
                for slot, value in test_env.item_dict_target.items():
                    if slot=='venuename':
                        continue                    
                    if int(slot) in set(inform_slots):
                        test_env.agent_request_slots.append(int(slot))
                        test_env.user_inform_fea_ids.extend(value)
                test_env.cand_items = list(range(test_env.item_length))  # candidate items
                test_env.agent_request_slots = []
                test_env.cur_conver_step = 1
                test_env.user_acc_feature = []  # user accepted value_feature which asked by agent
                test_env.user_rej_feature = [] 
                
        # next user    
        if test_env.dialog_task_ind==test_env.task_num and done==1:
            continue

    SR5_mean = np.mean(np.array([item[0] for item in result]))
    SR10_mean = np.mean(np.array([item[1] for item in result]))
    SR15_mean = np.mean(np.array([item[2] for item in result]))
    AvgT_mean = np.mean(np.array([item[3] for item in result]))
    SR_all = [SR5_mean, SR10_mean, SR15_mean, AvgT_mean]
    print('SR_all: ',SR_all)
    
    log.close()


def main():

    args = parameter_parser()
    torch.backends.cudnn.benchmark = True
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    ############################## dataloader act ##############################
    transformer_model = 'microsoft/DialoGPT-small'
    train = TextDataset.create_data('../code/dialogpt/resources/train.action_prediction', tokenizer_or_transformer_model=transformer_model, end_token=' <|endofaction|>', remove_tokens={}, split_token='<|endofbelief|>', split=(1,), shuffle=False)
    tokenizer = train.tokenizer
    pad_value = train.tokenizer.pad_token_id
    if pad_value is None:
        pad_value = train.tokenizer.eos_token_id
    cuda = True
    device_ids = [0]
    print('device_ids: ', device_ids)
    steps = 1000
    act_predict_model = DialoGPT(transformer_model, num_training_steps=steps, lr=5e-5, device_idxs=device_ids, cuda=cuda, pad_value=pad_value)
    print('loading the gpt checkpoint.............')
    
    act_predict_model.to('cuda:0')
    
    if args.act_single:
        act_predict_model.load_model('../code/dialogpt/checkpoint/microsoft#DialoGPT-small_11-01-2021 21:39:38.1610372378_val_acc-AR:0.166_val_loss-L:8.23 LR:8.234.th') 
    else:
        act_predict_model.load_model(
            '../code/dialogpt/checkpoint/action_prediction_microsoft#DialoGPT-small_21-03-2021 03:42:18.1616269338_val_acc-AR:0.37_val_loss-L:5.63 LR:5.625.th'
            )
    
    ############################## sgcn ##############################
    with_image = '_with_image' if args.with_image else '_wo_image'	
    act_single = '_act_single' if args.act_single else '_act_multi'
    hidden_dim = '_hidden_' + str(args.hidden_dim)
    lr = args.lr
    print('lr: ', lr)
    LR = '_lr_' + str(lr)
    if args.mlp:
        run_name = 'mlp' + with_image + act_single + hidden_dim +LR 
    else:
        run_name = with_image + act_single + hidden_dim +LR 
    print('run_name: ', run_name)

    checkpoint_dir = os.path.join('checkpoint_file/', run_name)
    train_data = MmdialData(args, mode = 'test')
    USER_SIZE = len(train_data.dial2user)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    SGR_model = SGR(args, device, USER_SIZE, train_data)
    SGR_model = SGR_model.to(device)

    start_epoch_id = args.step  
    modelWeight = torch.load(os.path.join(checkpoint_dir, 'checkpoint_'+ str(start_epoch_id)+'.tar'))['state_dict']  
    print('load checkpoint: ', start_epoch_id)
    SGR_model.load_state_dict(modelWeight) 
    for name,parameters in SGR_model.named_parameters():
        print(name,':',parameters.size())
    evaluate(args, device, train_data, act_predict_model, SGR_model, tokenizer, pad_value)

if __name__ == '__main__':
    main()