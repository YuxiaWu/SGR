# testing for different domain
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

#from data import MmdialData
from MmdialDataPre import MmdialData
from model import SGR
from runner import *
# from dialogpt folder
#from dataloader import TextDataset
#from model_rg import DialoGPT
#from model_utils import loop, get_device_ids, CustomPaddingTensorCollator, set_max_len
#from transformers import AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#print(args.gpu)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

run_name = 'mlp_neg_1_with_image_act_multi_hidden_128_lr_0.001_neg_1_wo_pretrain'
#run_name = 'img_sim_0.7_mlp_neg_1.0_with_image_act_multi_hidden_128_lr_0.001_neg_1.0_wo_pretrain'

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

start_epoch_id = 3  
modelWeight = torch.load(os.path.join(checkpoint_dir, 'checkpoint_'+ str(start_epoch_id)+'.tar'))['state_dict']  
print('load checkpoint: ', start_epoch_id)
model.load_state_dict(modelWeight) #		
print('testing .................................................')
testing_domain(args, writer, model, test_data, start_epoch_id, run_name, mode = 'test')	

writer.close()	