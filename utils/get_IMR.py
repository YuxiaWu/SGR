import torch
import time
import os
import json
import pickle
from param_parser import parameter_parser
from collections import Counter

args = parameter_parser()

with_image = '_with_image' if args.with_image else '_wo_image'
act_single = '_act_single' if args.act_single else '_act_multi'

run_name = with_image + act_single
print(run_name)
step_train = 300

with open(os.path.join('results', run_name, 'EMR_IMR_' + str(step_train)+'.pkl'),'rb') as f:
	EMR_IMR = pickle.load(f)

print(EMR_IMR)    

with open(os.path.join('results', run_name, 'item_rec_for_IMR_' + str(step_train)+'.pkl'),'rb') as f:
	item_rec = pickle.load(f)
    
    
with open(os.path.join('sample_for_train', with_image[1:], 'train_goal_id.json'),'r') as f:
	train_gt_item_id = json.load(f)    
with open(os.path.join('sample_for_train', with_image[1:], 'valid_goal_id.json'),'r') as f:
	val_gt_item_id = json.load(f)      
with open(os.path.join('sample_for_train', with_image[1:], 'test_goal_id.json'),'r') as f:
	test_gt_item_id = json.load(f)


result = Counter(train_gt_item_id)
#print(result)    
    
new_dct = {k: v for k, v in result.items() if v >8}


old, new = 0, 0
for test_gt_id in test_gt_item_id:
    
    if test_gt_id in new_dct.keys():
        old+=1
    else:
        new+=1
print(old)
print(new)


#train_gt_item_id = set([j for i in train_gt_item_id for j in i])





gt_item_id = {}
def _IMR_new_old(item_rec):
	IMR = []
	num=0
	new_IMR = []
	old_IMR = []
	for user_id, rec in item_rec.items():
		test_gt_id = test_gt_item_id[num] 
        
		gt_id = rec['gt_item_id']
		gt_item_id[user_id] = gt_id
		#print('len(gt_id)', len(gt_id))
		gt_id = list(set(gt_id))
		gt_id = list(range(len(gt_id)))
		gt_id = [str(i) for i in gt_id]
        
		'''
		if len(gt_id)>0:
            
			pre_id_top1 = rec['pre_item_id']['top1']
			pre_id_top1 = [str(int(eval(i))) for i in pre_id_top1]
			for gt_id, test_id in zip(gt_id_all, test_gt_id):
				imr_top1 = len(set(pre_id_top1) & set(gt_id))/len(gt_id)
				if test_id in new_dct.keys(): 
					old_IMR.append(torch.Tensor([imr_top1]))
				else:
					#print('new')
					new_IMR.append(torch.Tensor([imr_top1]))                    
        
		'''
        
		if len(gt_id)>0:
            
			pre_id_top1 = rec['pre_item_id']['top1']
			pre_id_top1 = [str(int(eval(i))) for i in pre_id_top1]          
			
			if len(pre_id_top1)>0:
				pre_id_top3 = rec['pre_item_id']['top3']
				pre_id_top5 = rec['pre_item_id']['top5']
				pre_id_top3 = [str(int(eval(i))) for i in pre_id_top3]
				pre_id_top5 = [str(int(eval(i))) for i in pre_id_top5]
                                
				imr_top1 = len(set(pre_id_top1) & set(gt_id))/len(gt_id)
				imr_top3 = len(set(pre_id_top3) & set(gt_id))/len(gt_id)
				imr_top5 = len(set(pre_id_top5) & set(gt_id))/len(gt_id)
				#print('gt_id: ', gt_id)
				#print('imr_5: ', pre_id_top5)
				#print(torch.Tensor([imr_top1,imr_top3,imr_top5]))
				if test_gt_id in new_dct.keys():
					old_IMR.append(torch.Tensor([imr_top1,imr_top3,imr_top5]))
				else:
					new_IMR.append(torch.Tensor([imr_top1,imr_top3,imr_top5]))
         
				IMR.append(torch.Tensor([imr_top1,imr_top3,imr_top5]))            
            
			

	return IMR, new_IMR, old_IMR


epoch_IMR, epoch_new_IMR, epoch_old_IMR = _IMR_new_old(item_rec)

IMR = torch.mean(torch.vstack(epoch_IMR),0)
print('IMR_size: ', torch.vstack(epoch_IMR).size())
print('IMR: ', IMR)



new_IMR = torch.mean(torch.vstack(epoch_new_IMR),0)
print('new_IMR_size: ', torch.vstack(epoch_new_IMR).size())
print('new_IMR: ', new_IMR)

old_IMR = torch.mean(torch.vstack(epoch_old_IMR),0)
print('old_IMR_size: ', torch.vstack(epoch_old_IMR).size())
print('old_IMR: ', old_IMR)


gt_item_id = {}
def _IMR(item_rec):
	IMR = []
	num=0
	for user_id, rec in item_rec.items():
		test_gt_id = test_gt_item_id[num]
        
           
		gt_id = rec['gt_item_id']
		gt_item_id[user_id] = gt_id
		print('len(gt_id)', len(gt_id))
		gt_id = list(set(gt_id))
		gt_id = list(range(len(gt_id)))
		gt_id = [str(i) for i in gt_id]
        
		if len(gt_id)>0:
			pre_id_top1 = rec['pre_item_id']['top1']
			pre_id_top1 = [str(int(eval(i))) for i in pre_id_top1]
			if len(pre_id_top1)>0:
				pre_id_top3 = rec['pre_item_id']['top3']
				pre_id_top5 = rec['pre_item_id']['top5']
				pre_id_top3 = [str(int(eval(i))) for i in pre_id_top3]
				pre_id_top5 = [str(int(eval(i))) for i in pre_id_top5]
                                
				imr_top1 = len(set(pre_id_top5[:len(gt_id)]) & set(gt_id))/len(gt_id)
				imr_top3 = len(set(pre_id_top3) & set(gt_id))/len(gt_id)
				imr_top5 = len(set(pre_id_top5) & set(gt_id))/len(gt_id)
				print('gt_id: ', gt_id)
				print('imr_5: ', pre_id_top5)
				print(torch.Tensor([imr_top1,imr_top3,imr_top5]))
				IMR.append(torch.Tensor([imr_top1,imr_top3,imr_top5]))
	return IMR


epoch_IMR = _IMR(item_rec)

IMR = torch.mean(torch.vstack(epoch_IMR),0)
print('IMR_size: ', torch.vstack(epoch_IMR).size())
print('IMR: ', IMR)


#with open('gt_item_id_test.json', 'w') as f:
#    json.dump(gt_item_id,f)
#with open('gt_item_id_test.pkl', 'wb') as f:
#    pickle.dump(gt_item_id,f)