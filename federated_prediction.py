#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
#Helper Source: https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/federated_main.py

import os
import sys
import copy
import time
import pickle
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('Agg')

import helper
import utils
import torch
from torch import nn
import torch.utils.data as data_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from local_model import LocalModel
from models import LSTM
from options import args_parser

#==========PARAMETERS=======================
global_epochs = 4 #number of epochs the global model will run
local_epochs = 5  #number of epochs each local model will run
num_hidden_nodes = 50 #number of hidden nodes in LSTM model 
frac = 0.7 #fraction of groups/meters to choose [**Note: currently not used. left for future] 
num_participants = 50  #number of participants to be applied to federated learning [**Note: set to None to use 'frac' of total participants]
split_ratio = 0.8 #split ratio for train data  [**Note: split ratio will only work when test_len is set to None]
test_len = 1440 # 30 days (30*48) of data  [**Note: set to None to split data based on a ratio]
normalize_data = True  #perform max-min normalization on the data
window_size = 48  #length of the input sequence to be used for predicting the next time step data point 
batch_size = 24 #16, 24, 48 :size of batch for training and testing, e.g. for total number of sequences 24,096 (24,144-48), having batch size 96 will produce 251 batches for training
take_all = True #whether federated learning will be applied to all participants
random_group = False # whether groups were formed by choosing meters randomly or in sorted order
gid = 'g9' #group id of the meters 
write_predictions = False
#===========================================

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If a GPU available, it will set device to GPU.
if is_cuda:
    device = torch.device("cuda:0")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

# define paths
base_path = os.getcwd()

#---------------------------------------------
def average_weights(w):
	"""
	Returns the average of the weights.
	"""
	w_avg = copy.deepcopy(w[0])
	for key in w_avg.keys():
		for i in range(1, len(w)):
			w_avg[key] += w[i][key]
		w_avg[key] = torch.div(w_avg[key], len(w))
    
	return w_avg

#----------------------------------------------
def global_inference(test_data, scaler, model):
	""" 
	Returns the inference and loss on global test data.
	"""

	x_test, y_test = utils.create_inout_sequences(test_data, window_size)
	test = data_utils.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
	test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=False)

	model.to(device)
	
	criterion = nn.MSELoss().to(device)

	model.eval()
	#print(next(model.parameters()).is_cuda)
	losses = []
	total_seq = 0
	predicted_values = []
	actual_values = []

	with torch.no_grad():
		for batch_idx, (seq, targets) in enumerate(test_loader):
			seq, targets = seq.to(device), targets.to(device)		
			targets = targets.view(-1, 1)
			#print(targets.shape)
			
			model.hidden_cell = model.init_hidden(device)
			outputs = model(seq)
			#print(outputs.shape)
			
			loss = criterion(outputs, targets)

			for out_tensor in outputs:
				predicted_values.append(out_tensor.item())
			#print(predicted_values[:5])
			for target_tensor in targets:
				actual_values.append(target_tensor.item())
			#print(actual_values[:5])				

	if normalize_data and scaler is not None:
		actual_values = scaler.inverse_transform(np.array(actual_values).reshape(-1, 1))
		predicted_values = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))
		
		predicted_values = predicted_values[:,0]
		actual_values = actual_values[:,0]      

	return actual_values, predicted_values, losses

#----------------------------------------------
def global_train_test(g_id):
	"""
	Returns train and test dataset for a given group/aggregator.
	"""
	group_type = "random_" if random_group else "" 
	dataframe = pd.read_csv(base_path + "/data/"+group_type+"group_load/"+str(g_id)+"_val"+".csv")
	
	all_data = dataframe['group_value'].values
	
	all_data = all_data.astype('float32')
	    
	if normalize_data:
		#perform min/max scaling on the dataset which normalizes the data within a certain range of minimum and maximum values. 
		scaler = MinMaxScaler(feature_range=(0, 1))
		all_data_normalized = scaler.fit_transform(all_data.reshape(-1, 1))
		#print(all_data_normalized)
		all_data = all_data_normalized
		

	# split into train and test sets (default: 80/20)
	if test_len is None:
		train_size = int(len(all_data) * split_ratio)
		test_size = len(all_data) - train_size
	else:
		train_size = len(all_data) - test_len		
		test_size = test_len
		
	train_data, test_data = all_data[:train_size], all_data[train_size:len(all_data)]
	if not normalize_data:
		train_data = train_data.reshape(-1, 1)
		test_data = test_data.reshape(-1, 1)
	
	return train_data, test_data, scaler

#--------------------------------------------
if __name__ == '__main__':
	start_time = time.time()
	args = args_parser()

	group_type = "random_" if random_group else "" 	
	#-------SET group ids from data folder~~~~~~~~~~~
	#groups = helper.find_filenames_ext(base_path + "/data/"+group_type+"group_load/")
	#group_ids = [ group[ : group.rindex("_")] for group in groups] 

	if args.group_id:
		gid = args.group_id
	else:
		print("Group ID needs to be specified")	
		sys.exit(0)

	if args.batch_size:
		batch_size = args.batch_size
	else:
		print("batch size needs to be specified")	
		sys.exit(0)		

	print("Starting federated learning for group %s"%gid)

	global_train, global_test, global_scaler = global_train_test(gid)
	global_test_max = np.amax(global_test)	
	global_test_min = np.amin(global_test)

	
	meter_ids = helper.read_txt(base_path + "/data/"+group_type+"group_ids/" + gid)
	#~~~~~~~~SET meter_ids from folder~~~~~~~~~~~~~~
	group_ids = [ int(meter) for meter in meter_ids ] #for this experiment each group corresponds to one meter 
	#~~~~~~~~SET meter/group ids manually~~~~~~~~~~~~
	#group_ids = [4820, 2826, 7370]
	#~~~~~~~~~~~~~~~~~TOTAL NUMBER OF PARTICIPANTS~~~~~~~~~~~~~~~~~~~~~~~~	
	num_groups = len(group_ids)	# **for this experiment num_groups = number of meters in a group 
	#num_groups = len(groups)
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# create model object
	global_model = LSTM(hidden_state_size=num_hidden_nodes, batch_first=True, batch_size=batch_size)
    
    # Set the model to train and send it to device.
	global_model.to(device)
	global_model.train()
    #print(global_model)

    # copy weights
	global_weights = global_model.state_dict()

    # Training
	train_loss = []
    
	#~~~~~~~~~~~~~~~~~CHOOSE ALL PARTICIPANTS~~~~~~~~~~~~~~~
	local_models = []	
	if take_all:		
		group_indices = np.arange(num_groups)
		for indx in group_indices:
			local_model = LocalModel(group_ids[indx], split_ratio, test_len, normalize=normalize_data, window=window_size, batch_size=batch_size, local_epochs=local_epochs, device=device)
			local_models.append(local_model)
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	for epoch in range(global_epochs):
		local_weights, local_losses = [], []
	
		global_model.train()
		#~~~~~~~~~~~~~~~~~CHOOSE FRACTION OF ALL PARTICIPANTS RANDOMLY IN EACH EPOCH~~~~~~~~~~~ 
		if not take_all:
			m = max(int(frac * num_groups), 1)
			if num_participants is None: 
				group_indices = np.random.choice(range(num_groups), m, replace=False)
			else:
				group_indices = np.random.choice(range(num_groups), num_participants, replace=False)
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		for indx in tqdm(group_indices):
			print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
			print("Training meter with ID: %s"%str(group_ids[indx]))
			#~~~~~~~~~~~~~~~when FRACTION OF ALL PARTICIPANTS are choosen randomly~~~~~~~~~~~~
			if not take_all:
				local_model = LocalModel(group_ids[indx], split_ratio, test_len, normalize=normalize_data, window=window_size, batch_size=batch_size, local_epochs=local_epochs, device=device)
				#local_models.append(local_model)
			#~~~~~~~~~~~~~~~~~when ALL PARTICIPANTS are chosen~~~~~~~~~~~~~~~~~~~~~~~~~~~
			else:			
				local_model = local_models[indx]
			#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch+1)
			local_weights.append(copy.deepcopy(w))
			local_losses.append(copy.deepcopy(loss))

        # update global weights
		global_weights = average_weights(local_weights)

        # update global weights
		global_model.load_state_dict(global_weights)

		loss_avg = sum(local_losses) / len(local_losses)
		train_loss.append({'epoch': epoch, 'locals_loss_avg': loss_avg})
		print("---------------------------------------------------")
		print('Global Training Round : {}, Average loss {:.3f}'.format(epoch+1, loss_avg))
		print("---------------------------------------------------")

	print('\nTotal Training Time: {0:0.4f}'.format(time.time()-start_time))
	#------------------------------------------------------------------------
	helper.make_dir(base_path, "results")
	helper.write_csv(base_path + "/results/federated-"+group_type+gid+"-train-avg-loss", train_loss, ["epoch", "locals_loss_avg"])

	#~~~~~~~~~~~~~~~when ALL PARTICIPANTS are chosen~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Inference on test data of local participants after completion of training
	print(f'\nResults of participants after {global_epochs} global rounds of training:')	
	group_losses = []
	
	rmse_list = []
	#global_model.eval()
	
	for indx in range(num_groups):
		meter_id = group_ids[indx]
		if not take_all:
			local_model = LocalModel(meter_id, split_ratio, normalize=normalize_data, window=window_size, batch_size=batch_size, local_epochs=local_epochs, device=device)
		else:
			local_model = local_models[indx]
		
		test_data_max = np.amax(local_model.test)	
		test_data_min = np.amin(local_model.test)

		act_values, pred_values, losses = local_model.inference(model=global_model)
		#act_values, pred_values, losses = local_model.inference(model=copy.deepcopy(global_model))

		helper.make_dir(base_path, "raw_results/federated/")
		if write_predictions:
			utils.write_predictions(base_path + "/raw_results/federated/"+ "federated-local-"+group_type+str(gid)+"-"+str(meter_id), act_values, pred_values)

		group_losses.append(losses)
		rmse = math.sqrt(mean_squared_error(act_values, pred_values))
		nrmse = rmse / (test_data_max - test_data_min)
		mae = mean_absolute_error(act_values, pred_values)	
		mape = utils.mean_absolute_percentage_error(act_values, pred_values)
	
		print("RMSE of meter %s: %.2f"%(str(group_ids[indx]), rmse))
		print('NRMSE of meter %s : %.2f' %(str(group_ids[indx]), nrmse))
		print('MAE of meter %s : %.2f' %(str(group_ids[indx]), mae))
		print("---------------------------------------------------")

		rmse_list.append({'meter_id': str(group_ids[indx]), 'RMSE': rmse, 'NRMSE': nrmse, 'MAE': mae, 'MAPE': mape})   	
 
	print('\nTotal Run Time: {0:0.4f}'.format(time.time()-start_time))
	
	helper.write_csv(base_path + "/results/federated-local-"+group_type+str(gid), rmse_list, ["meter_id", "RMSE", "NRMSE", "MAE", "MAPE"])

    #---------------------------------------------------------
	
	#~~~~~~~~~~~~~~~~~~~~Performance of Aggregator using global model trained in federated settings~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	global_metrics = []
	actual_values, predicted_values, test_losses = global_inference(global_test, global_scaler, global_model)

	helper.make_dir(base_path, "raw_results")
	if write_predictions:
		utils.write_predictions(base_path + "/raw_results/"+ "federated-global-"+group_type+str(gid), actual_values, predicted_values)

	rmse = math.sqrt(mean_squared_error(actual_values, predicted_values))
	nrmse = rmse / (global_test_max - global_test_min)
	mae = mean_absolute_error(actual_values, predicted_values)
	mape = utils.mean_absolute_percentage_error(actual_values, predicted_values)
	
	global_metrics.append({'group_id': gid, 'RMSE': rmse, 'NRMSE': nrmse, 'MAE': mae, 'MAPE': mape})

	print("Global RMSE of group %s: %.2f"%(gid, rmse))
	print('Global NRMSE of group %s : %.2f' %(gid, nrmse))
	print('Global MAE of group %s : %.2f' %(gid, mae))
	print("---------------------------------------------------")
	helper.write_csv(base_path + "/results/federated-global-"+group_type+str(gid), global_metrics, ["group_id", "RMSE", "NRMSE", "MAE", "MAPE"])
	
