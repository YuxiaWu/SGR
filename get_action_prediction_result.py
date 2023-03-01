
import sys
sys.path.append('../')
sys.path.append('dialogpt/')
sys.path.append('dialogpt/utils/')
sys.path.append('./utils/')

import numpy as np
import torch
import json
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time
from param_parser import parameter_parser
import os

print(sys.path)
from dataloader import TextDataset
from model_rg import DialoGPT
from model_utils import loop, get_device_ids, CustomPaddingTensorCollator, set_max_len
import random
from transformers import AutoTokenizer

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
args = parameter_parser()

transformer_model = './DialoGPT-small'

if args.act_single:
    input_format = 'action_prediction_single'
    result_path = os.path.join(args.data_path, 'act_prediction_result_single')
elif not args.with_image:
    input_format = 'action_prediction_woimage'
    result_path = os.path.join(args.data_path, 'act_prediction_result_woimage')

else:
    input_format = 'action_prediction'
    result_path = os.path.join(args.data_path, 'act_prediction_result')

if not os.path.exists(result_path):
    os.makedirs(result_path)
print('result_path: ', result_path)

train = TextDataset.create_data(f'./dialogpt/resources/train.{input_format}', tokenizer_or_transformer_model=transformer_model, end_token=' <|endofaction|>', remove_tokens={}, split_token='<|endofbelief|>', split=(1,), shuffle=False)
val = TextDataset.create_data(f'./dialogpt/resources/val.{input_format}', tokenizer_or_transformer_model=transformer_model, end_token=' <|endofaction|>', remove_tokens={}, split_token='<|endofbelief|>', split=(1,), training=False, shuffle=False)
test = TextDataset.create_data(f'./dialogpt/resources/test.{input_format}', tokenizer_or_transformer_model=transformer_model, end_token=' <|endofaction|>', remove_tokens={}, split_token='<|endofbelief|>', split=(1,), training=False, shuffle=False)

set_max_len(train, val, test)
pad_value = train.tokenizer.pad_token_id
if pad_value is None:
	pad_value = train.tokenizer.eos_token_id

key2pad_id = {
	'input_ids': pad_value,
	'labels': pad_value
}

first_eval = {
	'input_ids': True,
	'attention_mask': True
}

collator_train = CustomPaddingTensorCollator(key2pad_id=key2pad_id)
collator_eval = CustomPaddingTensorCollator(key2pad_id=key2pad_id, first=first_eval)

BATCH_SIZE = 1
dataloader_train_act = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, collate_fn=collator_train)
dataloader_val_act = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, collate_fn=collator_eval)
dataloader_test_act = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, drop_last=False, collate_fn=collator_eval)

dataloader_act_all = [dataloader_train_act, dataloader_val_act, dataloader_test_act]
device_ids = [0]
steps = 1000

act_predict_model = DialoGPT(transformer_model, num_training_steps=steps, lr=5e-5, device_idxs=device_ids, cuda=True, pad_value=pad_value)
print('loading the gpt checkpoint.............')

if args.act_single:
    act_predict_model.load_model('./dialogpt/checkpoint/action_prediction_single_microsoft#DialoGPT-small_02-08-2021 16:04:22.1627891462_val_acc-AR:0.55_val_loss-L:4.56 LR:4.562.th') 
elif not args.with_image:
    act_predict_model.load_model('./dialogpt/checkpoint/action_prediction_woimage_microsoft#DialoGPT-small_03-08-2021 01:57:20.1627927040_val_acc-AR:0.353_val_loss-L:6.16 LR:6.157.th') 
else:
    act_predict_model.load_model(
		'./dialogpt/checkpoint/action_prediction_microsoft#DialoGPT-small_13-01-2021 15:43:55.1610523835_val_acc-AR:0.371_val_loss-L:5.33 LR:5.325.th'
		)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
act_predict_model = act_predict_model.to(device)
act_predict_model.eval()
#  dialog id: turn id: [] 
tokenizer_or_transformer_model = transformer_model
if isinstance(tokenizer_or_transformer_model, str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_transformer_model)
else:
    tokenizer = tokenizer_or_transformer_model

def get_ap(act_predict_model, dataloader_act_all, mode):
    if mode=='train':
        dataloader_train_act = dataloader_act_all[0]        
    elif mode=='val':
        dataloader_train_act = dataloader_act_all[1] 
    else:
        dataloader_train_act = dataloader_act_all[2]        


    pre_action_all = []
    for bstate_seq in zip(dataloader_train_act):
        with torch.no_grad():      
            act_inputs, extras = act_predict_model.prepare(bstate_seq[0])
            outputs = act_predict_model(**act_inputs)
            results = act_predict_model.make_results(outputs, extras)


        for idx, result in results.items(): # only one
            result = result[0]
            pre = result['predictions']['rg']
            gt = result['gts']['rg']
            pre_token = tokenizer.convert_ids_to_tokens(pre)
            pre_action = [i[1:] for i in pre_token if i[1:] in set(['inform', 'request', 'recommend'])]
            if args.act_single:
                print('pre_action: ', pre_action)
                try:
                    pre_action = pre_action[0]
                except:
                    pass
            pre_action = sorted(pre_action)
            pre_action_all.append(pre_action)
    

    with open(os.path.join(result_path, mode + '.json'), 'w', encoding="utf-8") as f:
        f.write(json.dumps(pre_action_all, indent=4))          


start_time = time.time() 
get_ap(act_predict_model, dataloader_act_all, mode = 'train')
get_ap(act_predict_model, dataloader_act_all, mode = 'val')
get_ap(act_predict_model, dataloader_act_all, mode = 'test')

print('time cost: {}'.format(time.time()-start_time)) 
