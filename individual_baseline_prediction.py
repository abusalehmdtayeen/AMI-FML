#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
#Helper Source: https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
#--------------------------------------------------
# LSTM Implementation with Batch Training support
#----------------------------------------------------
import os
import time
import sys
import math
import torch
from torch import nn
import torch.utils.data as data_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from options import args_parser
from models import LSTM
import helper
import utils

#=====================================
base_path = os.getcwd()
#data_path = base_path + "/data/group_load/"
data_path = base_path + "/data/meter_load_half_hr/"
#============PARAMETERS===============
split_ratio = 0.80 #split ratio for train data
test_len = 1440 #number of test points= 30 days (30*48) of data  [**Note: set to None to split data based on a ratio]
normalize = True #whether to perform max-min normalization on the data
num_hidden_nodes = 50 #number of hidden nodes in LSTM model 
window_size = 48 #sequence length: length of the input sequence to be used for predicting the next time step data point 
batch_size = 24 #16, 24, 48 #size of batch for training and testing, e.g. for total number of sequences 24,096 (24,144-48), having batch size 96 will produce 251 batches for training
epochs = 5 #number of epochs each model will run
random_group = False # whether groups were formed by choosing meters randomly or in sorted order
#gid = 'g9' #group id of the meters used as command line parameter
write_predictions = True
#====================================

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU.
if is_cuda:
    device = torch.device("cuda:0")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

#----------------------------------------
# load the dataset
def get_train_test_data(group_id):
	"""
	The function to get train and test data. 
  
        Parameters: 
            test_len (int): The number of data points to test. 
          
        Returns: 
	"""
	print("Getting data of meter with ID: %s"%str(group_id))
	#dataframe = pd.read_csv(data_path+str(group_id)+"_sum"+".csv")
	dataframe = pd.read_csv(data_path+str(group_id)+".csv")
	#print(dataframe.shape)
	#print(dataframe.head())
	#plot_dataset(dataframe)

	#all_data = dataframe['group_sum'].values
	all_data = dataframe['load'].values
	all_data = all_data.astype('float32')
	print("Total number of timesteps: %d"%all_data.shape[0])
	#print(all_data)

	scaler = None
	if normalize:
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
	#train_data = all_data[:-test_data_size]
	print("Total train timesteps: %d"%len(train_data))
	#print(train_data)
	#test_data = all_data[-test_data_size:]
	if not normalize:
		train_data = train_data.reshape(-1, 1)
		test_data = test_data.reshape(-1, 1)
	
	print("Total test timesteps: %d"%len(test_data))
	#print(test_data)
	return train_data, test_data, scaler

#---------------------------------------
def train_model(model, train_data):
	x_train, y_train = utils.create_inout_sequences(train_data, window_size)
	train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
	train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=False)

	print("Number of train batches: %d"%len(train_loader))

	loss_function = nn.MSELoss().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

	model.train()
	print("Started training the model...")
	start_time = time.time()

	for i in range(epochs):
		batch_loss = 0.
		for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
			#print (x_batch.shape)
			seq, labels = x_batch.to(device), y_batch.to(device)
			labels = labels.view(-1, 1)
			optimizer.zero_grad()
			model.hidden_cell = model.init_hidden(batch_size, device)
			y_pred = model(seq)
			
			loss = loss_function(y_pred, labels)
			loss.backward()
			optimizer.step()

			batch_loss += loss.item()

		avg_loss = batch_loss / (batch_idx+1)
		if i%1 == 0:
			print(f'Epoch: {i:3} Avg loss: {avg_loss:10.8f}')

	elapsed_time = time.time() - start_time
	print(f'Training time: {elapsed_time: 10.4f}')

	return model
	
#-------------------------------------
def mean_absolute_percentage_error(y_t, y_p):
	y_t, y_p = np.array(y_t), np.array(y_p)
	
	try:
		mape = np.mean(np.abs((y_t-y_p)/y_t))
	except:
		mape = -1.0

	return mape

#-----------------------------------
#evaluate the model with test data
def evaluate_model(model, group_id, test_data, scaler=None):
	test_data_max = np.amax(test_data)	
	test_data_min = np.amin(test_data)

	x_test, y_test = utils.create_inout_sequences(test_data, window_size)
	test = data_utils.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
	#print(test[:-12])
	test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=False)
	print("Number of test batches: %d"%len(test_loader))

	criterion = nn.MSELoss().to(device)

	predicted_values = []
	actual_values = []
	
	model.eval()
	print("Evaluating the model...")
	with torch.no_grad():
		for batch_idx, (seq, targets) in enumerate(test_loader):
			seq, targets = seq.to(device), targets.to(device)		
			targets = targets.view(-1, 1)
			#print(targets.shape)
			
			model.hidden_cell = model.init_hidden(batch_size, device)
			outputs = model(seq)
			#print(outputs.shape)
			
			loss = criterion(outputs, targets)

			for out_tensor in outputs:
				predicted_values.append(out_tensor.item())
			#print(predicted_values[:5])
			for target_tensor in targets:
				actual_values.append(target_tensor.item())
			#print(actual_values[:5])				

	if normalize and scaler is not None:
		#Since we normalized the dataset, the predicted values are also normalized. We need to convert the normalized predicted values into actual predicted values.
		actual_values = scaler.inverse_transform(np.array(actual_values).reshape(-1, 1))
		predicted_values = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))
		
		predicted_values = predicted_values[:,0]
		actual_values = actual_values[:,0]

	#print(actual_values[:5])
	#print("-----------------")
	#print(predicted_values[:5])	
	#print("-----------------")
	meter_id = group_id 
	helper.make_dir(base_path, "raw_results")
	if write_predictions:
		utils.write_predictions(base_path + "/raw_results/"+ "single-local-"+group_type+str(gid)+"-"+str(meter_id), actual_values, predicted_values)

	
	RMSE = math.sqrt(mean_squared_error(actual_values, predicted_values))
	NRMSE = RMSE / (test_data_max - test_data_min)
	MAE = mean_absolute_error(actual_values, predicted_values)
	MAPE = mean_absolute_percentage_error(actual_values, predicted_values)

	print('RMSE of meter with ID %s : %.2f' %(str(group_id), RMSE))
	print('NRMSE of meter with ID %s : %.2f' %(str(group_id), NRMSE))
	print('MAE of meter with ID %s : %.2f' %(str(group_id), MAE))
	print('MAPE of meter with ID %s : %.2f' %(str(group_id), MAPE))

	return RMSE, NRMSE, MAE, MAPE

#-------------------------------
if __name__ == '__main__':
	start_time = time.time()
	args = args_parser()
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

	print("Starting individual baseline prediction for group %s"%gid)
	

	groups = helper.find_filenames_ext(data_path)
	group_type = "random_" if random_group else "" 
	#~~~~~~~~SET group ids manually~~~~~~~~~~~~~~~
	#group_ids = ['7341', 'g2', 'g3'] #group ids of the meter clusters
	#-------SET group ids from data folder~~~~~~~~~~~
	#group_ids = [ group[ : group.rindex("_")] for group in groups]
	group_ids = helper.read_txt(base_path+"/data/"+group_type+"group_ids/"+str(gid)) 

	rmse_list = []
	for mid in tqdm(group_ids):
		print("--------------------------------------")
		train_data, test_data, scaler = get_train_test_data(mid)

		model = LSTM(hidden_state_size=num_hidden_nodes, batch_first=True, batch_size=batch_size)
		model.to(device)
		#print(model)

		model = train_model(model, train_data)
		rmse, nrmse, mae, mape = evaluate_model(model, mid, test_data, scaler)
		rmse_list.append({'meter_id': mid, 'rmse': rmse, 'nrmse': nrmse, 'mae': mae, 'mape': mape})
		
	end_time = time.time()
	print('\nTotal Run Time: {0:0.4f}'.format(end_time-start_time))
	helper.make_dir(base_path, "results")	
	helper.write_csv(base_path + "/results/single-meter-"+group_type+str(gid), rmse_list, ["meter_id", "rmse", "nrmse", "mae", "mape"])
