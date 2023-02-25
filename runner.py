import torch
import time
import os
import json
import pickle
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
def training(
	args, 
	writer, 
	model, 
	train_data, 
	run_name, 
	start_step_id,  
	checkpoint_dir,
	device
	):


	log_path = './log/log_train'
	if not os.path.exists(log_path):
		os.makedirs(log_path)
	log = open(os.path.join(log_path,run_name + ".txt"), "a")
	sys.stdout = log

	mode = 'train'
	model.train()
	num_epoch = args.num_epoch
	dataloader_train_SGR = DataLoader(train_data, batch_size=1,shuffle=True,drop_last=True,num_workers=4, pin_memory=True)
	num_batch = int(len(dataloader_train_SGR) / args.batch_size)
	print('num_batch: ', num_batch)
	print('start training model...')
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	for epoch in range(num_epoch):
		
		print('######################################################')
		epoch += int(start_step_id) 
		epoch_loss = 0
		start_epoch_time = time.time()	
		start_batch_time = time.time()	
		step = 0
		dataloader_time = time.time()
		print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
		sample_num = 0
		optimizer.zero_grad()
		for batch_SGR in dataloader_train_SGR:
			log.flush()
			sample_num +=1
			# start_sample_time = time.time()
			# bstate_pos, bstate_neg, agent, user_id,  act_pre
			batch_loss = model.train_one_sample(
				batch_SGR[0], 
				batch_SGR[1], 
				batch_SGR[2], 
				batch_SGR[3],
				batch_SGR[5])
			#print('sample time: ', time.time()-start_sample_time)
			batch_loss = batch_loss/args.batch_size
			epoch_loss = epoch_loss + batch_loss.item()				
			batch_loss.backward()
			del batch_loss
			#print(sample_num)
			if sample_num % args.batch_size==0:
				optimizer.step()
				optimizer.zero_grad()
				step+=1	
				#print(step_num)			
				if step % 10 == 0:
					print(
						'epoch: {}, step: {}, num_batch: {}, time cost:{:.4f}, loss: {:.4f}, time: {}'
						.format(
							epoch, step, num_batch, time.time()-start_batch_time, epoch_loss/step, 
							time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
							) 
						)
					start_batch_time = time.time()
				
		if epoch % args.save_every_epochs==0:
			savedir = os.path.join(checkpoint_dir) 
			if not os.path.exists(savedir):	
				os.makedirs(savedir)
			savename = savedir + "/checkpoint" + "_" + str(epoch) + ".tar"  
			torch.save({"epoch": epoch, "state_dict": model.state_dict()}, savename)	
		print('save_model done')

		print('epoch_loss: {:.4f}'.format(epoch_loss/step))
		print('epoch_time minutes: {:.2f}'.format((time.time()-start_epoch_time)/60))
		writer.add_scalar(mode + '/Loss', float('{:.4f}'.format(epoch_loss/step)), epoch)
	log.close()

def print_metrics(batch_EMR, batch_IMR, writer, mode, epoch):
	mean_batch_EMR = torch.mean(torch.vstack(batch_EMR),0)
	print('EMR@1: {:.4f}\t EMR@3: {:.4f}\t EMR@5: {:.4f}'.format(mean_batch_EMR[0], mean_batch_EMR[1],mean_batch_EMR[2]))
							
	mean_batch_IMR = torch.mean(torch.vstack(batch_IMR),0)
	print('IMR@1: {:.4f}\t IMR@3: {:.4f}\t IEMR@5: {:.4f}\t '
	.format(mean_batch_IMR[0], mean_batch_IMR[1],mean_batch_IMR[2]))
		 
	writer.add_scalar( mode + '/IMR@1 ', mean_batch_IMR[0], epoch)
	writer.add_scalar( mode +  '/IMR@3 ', mean_batch_IMR[1], epoch)
	writer.add_scalar( mode +  '/IMR@5 ', mean_batch_IMR[2], epoch)

def testing(args, writer, model, test_data, epoch_train, run_name, mode):
	
	#log_name = "testing_epoch_" + str(epoch_train)
	log_path = './log/log_test'
	if not os.path.exists(log_path):
		os.makedirs(log_path)
	log = open(os.path.join(log_path,  '1125epoch_' + str(epoch_train)  + run_name + ".txt"), "a")
	sys.stdout = log
	'''
	result_path = os.path.join('results', run_name)
	try:
		os.makedirs(result_path)	
	except:
		pass  
	'''
      
	#model.eval()
	dataloader_test_SGR = DataLoader(test_data, batch_size=1,shuffle=False,drop_last=False,num_workers=4, pin_memory=True)
	print('test_size: ', len(dataloader_test_SGR)) 

	#ith torch.no_grad():
	start_time = time.time()
	if True:
		epoch_loss = 0
		epoch_EMR = []
		epoch_inform_EMR, epoch_request_EMR, epoch_recommend_EMR = [],[],[]
		start_time = time.time()
		step_all = 0
		item_rec_for_IMR = {}
		 
		for batch_SGR in dataloader_test_SGR:
			log.flush()
			step_all+=1
			batch_graph_pos = batch_SGR[0]
			batch_graph_neg = batch_SGR[1]
			batch_agent = batch_SGR[2]
			batch_user = batch_SGR[3]
			batch_gt = batch_SGR[4]
			batch_act_pre = batch_SGR[5]
			batch_request_hist = batch_SGR[6]
			
			uid = str(batch_user.numpy()[0])
			if uid not in item_rec_for_IMR.keys():
				item_rec_for_IMR[uid] = {}
				item_rec_for_IMR[uid]['gt_item_id'] = []
				item_rec_for_IMR[uid]['pre_item_id'] = []
                
			batch_loss, batch_EMR, item_rec_for_IMR, batch_three_EMR  = model.inference_one_sample(
				model, batch_graph_pos, batch_graph_neg, batch_act_pre, batch_agent, batch_user, batch_gt, batch_request_hist, item_rec_for_IMR, epoch_train)
			epoch_loss = epoch_loss + batch_loss
			#print('epoch_loss: ',epoch_loss/step_all)
			#torch.cuda.empty_cache()
			#print('batch_EMR: ', batch_EMR)
			if len(batch_EMR)>0:
				epoch_EMR.append(torch.mean(torch.Tensor(batch_EMR),0))

				print('*****************************************************************')
				print('run_name: ', run_name)
				print('epoch_train: ', epoch_train)
				print('step_all_test: ', step_all)
				print('epoch_loss: ', epoch_loss/step_all)
				#print('epoch_EMR: ', torch.vstack(epoch_EMR))
				print('EMR size: ', torch.vstack(epoch_EMR).size())
				print('EMR: ', torch.mean(torch.vstack(epoch_EMR),0))


			else:
				#print('pre_action!=gt_action, EMR_metrics is []')
				pass 
			
			if len(batch_three_EMR[0])>0:
				epoch_inform_EMR.append(torch.mean(torch.Tensor(batch_three_EMR[0]),0))
				print('inform EMR size: ', torch.vstack(epoch_inform_EMR).size())
				
				print('inform_EMR: ', torch.mean(torch.vstack(epoch_inform_EMR),0))
			
			if len(batch_three_EMR[1])>0:
				epoch_request_EMR.append(torch.mean(torch.Tensor(batch_three_EMR[1]),0))
				print('request EMR size: ', torch.vstack(epoch_request_EMR).size())
				
				print('request_EMR: ', torch.mean(torch.vstack(epoch_request_EMR),0))
			
			if len(batch_three_EMR[2])>0:
				epoch_recommend_EMR.append(torch.mean(torch.Tensor(batch_three_EMR[2]),0))
				print('recommend EMR size: ', torch.vstack(epoch_recommend_EMR).size())
				
				print('recommend_EMR: ', torch.mean(torch.vstack(epoch_recommend_EMR),0))
			
		print('inform EMR size: ', torch.vstack(epoch_inform_EMR).size())
		print('inform_EMR: ', torch.mean(torch.vstack(epoch_inform_EMR),0))		
		print('request EMR size: ', torch.vstack(epoch_request_EMR).size())
		print('request_EMR: ', torch.mean(torch.vstack(epoch_request_EMR),0))
		print('recommend EMR size: ', torch.vstack(epoch_recommend_EMR).size())
		print('recommend_EMR: ', torch.mean(torch.vstack(epoch_recommend_EMR),0))
		
		epoch_IMR = get_IMR(item_rec_for_IMR)
		EMR = torch.mean(torch.vstack(epoch_EMR),0)
		IMR = torch.mean(torch.vstack(epoch_IMR),0)
		print('IMR_size: ', torch.vstack(epoch_IMR).size())
		print('IMR: ', IMR)
		print('time cost for testing: ', time.time()-start_time)

		'''
		with open(os.path.join(result_path,'item_rec_for_IMR_'+str(epoch_train)+'.json'),'wb') as f:
			json.dump(item_rec_for_IMR, f)
		with open(os.path.join(result_path,'EMR_IMR_'+ str(epoch_train) +'.json'),'wb') as f:
			json.dump([EMR, IMR], f)  
		writer.add_scalar( mode+ '/Loss ', epoch_loss/step_all, epoch_train)
		#writer.add_scalar( mode+ '/EMR ', EMR, epoch_train)
		#writer.add_scalar( mode+ '/IMR ', IMR, epoch_train)
        
		print('step: {}, loss: {:.5f}, cost_time: {:.1f}s'
			.format(epoch_train, epoch_loss/step_all, time.time() - start_time))    
		print_metrics(epoch_EMR, IMR, writer, mode, epoch_train)
		'''
		print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))		
	
	
	log.close()

def get_IMR(item_rec_for_IMR):
	IMR = []    
	for user_id, rec in item_rec_for_IMR.items():
		gt_id = rec['gt_item_id']
		if len(gt_id)>0:
			pre_id_top1 = rec['pre_item_id']
			if len(pre_id_top1)>0:
				imr_top1 = len(set(pre_id_top1) & set(gt_id))/len(gt_id)
				IMR.append(torch.Tensor([imr_top1]))
	return IMR

def get_IMR_domain(item_rec_for_IMR, uid_domain):
	IMR = []   
	IMR_domain = {'food': [], 'hotel': [],
				'nightlife':[], 'shopping mall':[], 'sightseeing':[] }	 
	for user_id, rec in item_rec_for_IMR.items():
		gt_id = rec['gt_item_id']
		if len(gt_id)>0:
			pre_id_top1 = rec['pre_item_id']
			if len(pre_id_top1)>0:
				imr_top1 = len(set(pre_id_top1) & set(gt_id))/len(gt_id)
				IMR.append(torch.Tensor([imr_top1]))
				domain = uid_domain[user_id]
				IMR_domain[domain].append(torch.Tensor([imr_top1]))
	return IMR, IMR_domain

def testing_domain(args, writer, model, test_data, epoch_train, run_name, mode):
	
	#log_name = "testing_epoch_" + str(epoch_train)
	log_path = './log/log_test_domain_1225_2'
	if not os.path.exists(log_path):
		os.makedirs(log_path)
	log = open(os.path.join(log_path,  'epoch_' + str(epoch_train)  + run_name + ".txt"), "w")
	sys.stdout = log
	'''
	result_path = os.path.join('results', run_name)
	try:
		os.makedirs(result_path)	
	except:
		pass  
	'''
	with open(os.path.join(args.data_path,'uid_domain.json'), "r", encoding="utf-8") as f:
		uid_domain = json.load(f)   
	#model.eval()
	dataloader_test_SGR = DataLoader(test_data, batch_size=1,shuffle=False,drop_last=False,num_workers=4, pin_memory=True)
	print('test_size: ', len(dataloader_test_SGR)) 

	#ith torch.no_grad():
	start_time = time.time()
	if True:
		epoch_loss = 0
		epoch_EMR = []
		epoch_EMR_domain = {'food': [], 'hotel': [],
				'nightlife':[], 'shopping mall':[], 'sightseeing':[] }
		epoch_inform_EMR, epoch_request_EMR, epoch_recommend_EMR = [],[],[]
		start_time = time.time()
		step_all = 0
		item_rec_for_IMR = {}
		step_domain = {'food': 0, 'hotel': 0,
				'nightlife':0, 'shopping mall':0, 'sightseeing':0 }
		 
		for batch_SGR in dataloader_test_SGR:
			log.flush()
			step_all+=1
			batch_graph_pos = batch_SGR[0]
			batch_graph_neg = batch_SGR[1]
			batch_agent = batch_SGR[2]
			batch_user = batch_SGR[3]
			batch_gt = batch_SGR[4]
			batch_act_pre = batch_SGR[5]
			batch_request_hist = batch_SGR[6]
			
			uid = str(batch_user.numpy()[0])
			if uid not in item_rec_for_IMR.keys():
				item_rec_for_IMR[uid] = {}
				item_rec_for_IMR[uid]['gt_item_id'] = []
				item_rec_for_IMR[uid]['pre_item_id'] = []
                
			batch_loss, batch_EMR, item_rec_for_IMR, batch_three_EMR  = model.inference_one_sample(
				model, batch_graph_pos, batch_graph_neg, batch_act_pre, batch_agent, batch_user, batch_gt, batch_request_hist, item_rec_for_IMR, epoch_train)
			epoch_loss = epoch_loss + batch_loss

			domain = uid_domain[uid]
			step_domain[domain]+=1

			if len(batch_EMR)>0:
				epoch_EMR.append(torch.mean(torch.Tensor(batch_EMR),0))
				epoch_EMR_domain[domain].append(torch.mean(torch.Tensor(batch_EMR),0))

				print('*****************************************************************')
				print('run_name: ', run_name)
				print('epoch_train: ', epoch_train)
				print('step_all_test: ', step_all)
				print('epoch_loss: ', epoch_loss/step_all)
				#print('epoch_EMR: ', torch.vstack(epoch_EMR))
				print('EMR size: ', torch.vstack(epoch_EMR).size())
				print('EMR_mean: ', torch.mean(torch.vstack(epoch_EMR),0))
				print('EMR: ', torch.sum(torch.vstack(epoch_EMR),0)/step_all)

				for domain, EMR_domain in epoch_EMR_domain.items():
					print('domain: ', domain)
					try:
						print('EMR_size: ', torch.vstack(EMR_domain).size())
					except:
						pass
					try:
						EMR_i = torch.mean(torch.vstack(EMR_domain),0)
						print('EMR_mean: ', EMR_i)
					except:
						pass
					try:
						EMR_i = torch.sum(torch.vstack(EMR_domain),0)/step_domain[domain]
						print('EMR: ', EMR_i)
					except:
						pass

			else:
				#print('pre_action!=gt_action, EMR_metrics is []')
				pass 
			
			if len(batch_three_EMR[0])>0:
				epoch_inform_EMR.append(torch.mean(torch.Tensor(batch_three_EMR[0]),0))
				print('inform EMR size: ', torch.vstack(epoch_inform_EMR).size())
				
				print('inform_EMR: ', torch.mean(torch.vstack(epoch_inform_EMR),0))
			
			if len(batch_three_EMR[1])>0:
				epoch_request_EMR.append(torch.mean(torch.Tensor(batch_three_EMR[1]),0))
				print('request EMR size: ', torch.vstack(epoch_request_EMR).size())
				
				print('request_EMR: ', torch.mean(torch.vstack(epoch_request_EMR),0))
			
			if len(batch_three_EMR[2])>0:
				epoch_recommend_EMR.append(torch.mean(torch.Tensor(batch_three_EMR[2]),0))
				print('recommend EMR size: ', torch.vstack(epoch_recommend_EMR).size())
				
				print('recommend_EMR: ', torch.mean(torch.vstack(epoch_recommend_EMR),0))
			
		print('inform EMR size: ', torch.vstack(epoch_inform_EMR).size())
		print('inform_EMR: ', torch.mean(torch.vstack(epoch_inform_EMR),0))		
		print('request EMR size: ', torch.vstack(epoch_request_EMR).size())
		print('request_EMR: ', torch.mean(torch.vstack(epoch_request_EMR),0))
		print('recommend EMR size: ', torch.vstack(epoch_recommend_EMR).size())
		print('recommend_EMR: ', torch.mean(torch.vstack(epoch_recommend_EMR),0))
		
		#epoch_IMR = get_IMR(item_rec_for_IMR)
		epoch_IMR, IMR_domain = get_IMR_domain(item_rec_for_IMR, uid_domain)
		EMR = torch.mean(torch.vstack(epoch_EMR),0)
		IMR = torch.mean(torch.vstack(epoch_IMR),0)
		print('IMR_size: ', torch.vstack(epoch_IMR).size())
		print('IMR_mean: ', IMR)

		IMR_i = torch.sum(torch.vstack(epoch_IMR),0)/1000
		print('IMR_sum/1000: ', IMR_i)
		# domain
		print('----domain-----')

		for domain, EMR_domain in epoch_EMR_domain.items():
			print('domain: ', domain)
			print('EMR_size: ', torch.vstack(EMR_domain).size())
			EMR_i = torch.mean(torch.vstack(EMR_domain),0)
			print('EMR_mean: ', EMR_i)
			EMR_i = torch.sum(torch.vstack(EMR_domain),0)/step_domain[domain]
			print('EMR: ', EMR_i)
		for domain, IMR_ in IMR_domain.items():
			print('domain: ', domain)
			print('IMR_size: ', torch.vstack(IMR_domain).size())
			IMR_i = torch.mean(torch.vstack(IMR_),0)
			print('IMR_mean: ', IMR_i)
			IMR_i = torch.sum(torch.vstack(IMR_),0)
			print('IMR_sum: ', IMR_i)
		print('time cost for testing: ', time.time()-start_time)
		print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))		
	log.close()