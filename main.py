import sys
sys.path.append('../')
sys.path.append('./utils')

#sys.path.append('../dialogpt/')
#sys.path.append('../dialogpt/utils/')
print(sys.path)

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
import time
from tensorboardX import SummaryWriter
from tqdm import trange
import random

from param_parser import parameter_parser

args = parameter_parser()
torch.backends.cudnn.benchmark = True
SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)


from MmdialDataPre import MmdialData
from model import SGR
from runner import *


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#print(args.gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)
lr = args.lr
print('lr: ', lr)
img_sim = 'img_sim_' + str(args.img_sim_thr)

with_image = '_with_image' if args.with_image else '_wo_image'	
act_single = '_act_single' if args.act_single else '_act_multi'
hidden_dim = '_hidden_' + str(args.hidden_dim)
neglink = '_neg_' + str(args.neglink_num)
LR = '_lr_' + str(lr)

run_name = img_sim
if args.mlp:
    run_name += '_mlp' + neglink + with_image + act_single + hidden_dim + LR + neglink
else:
    run_name += neglink + with_image + act_single + hidden_dim + LR 
if not args.pre_train:
	run_name += '_wo_pretrain'


if args.test_GCN:
	print('test_GCN_'+ str(args.test_GCN_epoch))

print('run_name: ', run_name)

checkpoint_dir = os.path.join('checkpoint_file/', run_name)
tensorboard_dir = os.path.join('tensorboard/', run_name )
try:
	os.makedirs(checkpoint_dir)	
	os.makedirs(tensorboard_dir)	
except:
	pass
writer = SummaryWriter(tensorboard_dir)

print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#print('args: ', args)
############################## dataloader score ##############################

train_data = MmdialData(args, mode = 'train')
val_data = MmdialData(args, mode = 'val')
test_data = MmdialData(args, mode = 'test')

USER_SIZE = len(train_data.dial2user)

############################## model ##############################
model = SGR(args, device, USER_SIZE, train_data)
model = model.to(device)

for name,parameters in model.named_parameters():
	print(name,':',parameters.size()) 

if args.load_model:
	start_epoch_id = args.step  
	modelWeight = torch.load(os.path.join(checkpoint_dir, 'checkpoint_'+ str(start_epoch_id)+'.tar'))['state_dict']  
	print('load checkpoint: ', start_epoch_id)
	model.load_state_dict(modelWeight) #		

	if args.test:
		print('testing .................................................')
		testing(args, writer, model, test_data, start_epoch_id, run_name, mode = 'test')	
	else:
		print('finetuning the model .................................................')
		training(args, writer, model, train_data, run_name, int(start_epoch_id)+1, checkpoint_dir, device)
		print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
else:
	start_epoch_id = -1
	print('training.................................................')
	training(args, writer, model, train_data, run_name, int(start_epoch_id)+1, checkpoint_dir, device)
	print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
writer.close()			
