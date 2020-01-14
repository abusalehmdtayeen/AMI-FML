#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
#Helper Source: https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/

import os
import sys
import math
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from models import LSTM
import helper
import utils

#=====================================
base_path = os.getcwd()
data_path = base_path + "/data/group_load/"
#============PARAMETERS===============
group_id = 3 #group id of the meter cluster
split_ratio = .80 #split ratio for train data
window_size = 48
epochs = 3
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
def get_train_test_data(normalize=True):
	print("Getting data of group with ID: %s"%str(group_id))
	dataframe = pd.read_csv(data_path+"g"+str(group_id)+"_sum"+".csv")
	#print(dataframe.shape)
	#print(dataframe.head())
	#plot_dataset(dataframe)

	all_data = dataframe['group_sum'].values
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
	train_size = int(len(all_data) * split_ratio)
	test_size = len(all_data) - train_size
		
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
	#convert train data into tensors since PyTorch models are trained using tensors
	train_data = torch.FloatTensor(train_data).view(-1)
	#print(train_data)

	#convert our training data into sequences and corresponding labels.
	train_inout_seq = utils.create_inout_sequences(train_data, window_size)
	#print(train_inout_seq[:5])

	model.train()
	print("Started training the model...")
	for i in range(epochs):
		for seq, labels in train_inout_seq:
			seq, labels = seq.to(device), labels.to(device)
			optimizer.zero_grad()
			model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size, device=device), torch.zeros(1, 1, model.hidden_layer_size, device=device))
			y_pred = model(seq)

			single_loss = loss_function(y_pred, labels)
			single_loss.backward()
			optimizer.step()

		print(f'epoch:{i:3} loss: {single_loss.item():10.8f}')

	#print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

#-----------------------------------
#evaluate the model with test data
def evaluate_model(model, test_data, normalized=True, scaler=None):
	test_data = torch.FloatTensor(test_data).view(-1)	
	test_seq = utils.create_inout_sequences(test_data, window_size)
	#print (test_seq)
	#print(len(test_seq))

	criterion = nn.MSELoss().to(device)

	test_predictions = []
	actual_predictions = []
	losses = []
	model.eval()
	print("Evaluating the model...")
	for seq, labels in test_seq:
		seq, labels = seq.to(device), labels.to(device)
		with torch.no_grad():
			model.hidden = (torch.zeros(1, 1, model.hidden_layer_size, device=device), torch.zeros(1, 1, model.hidden_layer_size, device=device))
			outputs = model(seq)
			loss = criterion(outputs, labels)
			losses.append(loss.item())
			test_predictions.append(outputs.item())
			actual_predictions.append(labels.item())

	#print(test_predictions)
	#print(actual_predictions)
	if normalized and scaler is not None:
		#Since we normalized the dataset, the predicted values are also normalized. We need to convert the normalized predicted values into actual predicted values.
		actual_predictions = scaler.inverse_transform(np.array(actual_predictions).reshape(-1, 1))
		test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
		
		test_predictions = test_predictions[:,0]
		actual_predictions = actual_predictions[:,0]

	#print(actual_predictions[:5])
	#print("-----------------")
	#print(test_predictions[:5])	
	#print("-----------------")
	helper.make_dir(base_path, "figures")
	utils.plot_predictions(base_path + "/figures/", group_id, actual_predictions, test_predictions)

	errors = [(i - j)**2 for i, j in zip(actual_predictions, test_predictions)] 
	#print(errors[: 5])
	#print(losses[: 5])
	utils.plot_errors(base_path + "/figures/", group_id, errors)
	utils.plot_losses(base_path + "/figures/", group_id, losses, lid="single")

	testScore = math.sqrt(mean_squared_error(actual_predictions, test_predictions))
	print('Test Score: %.2f RMSE' % (testScore))

#-------------------------------
if __name__ == '__main__':
	normalize = True
	train_data, test_data, scaler = get_train_test_data(normalize)
	
	model = LSTM()
	model.to(device)
	loss_function = nn.MSELoss().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	#print(model)

	train_model(model, train_data)
	evaluate_model(model, test_data, normalize, scaler)

