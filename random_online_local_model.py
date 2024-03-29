#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import math
import torch
import numpy as np
import pandas as pd
import utils
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch.utils.data as data_utils

#=============================================
class LocalModel(object):

	def __init__(self, gid, split_ratio=0.8, test_len=None, test_range=1, normalize=True, window = 24, batch_size=24, local_epochs=10, test_epochs=1, per_zeros=0.1, device=torch.device("cpu")):
		#self.data_path = os.getcwd() + "/data/group_load/"
		self.data_path = os.getcwd() + "/data/meter_load_half_hr/"
		self.normalize = normalize
		self.scaler = None
		self.test_len = test_len
		self.per_zeros = per_zeros #percentage of zeros in random mask
		self.batch_size = batch_size
		self.test_range = test_range # the number of test points to predict before updating the model
		self.test_epochs = test_epochs
		self.train, self.test = self.train_test_data(gid, split_ratio)
		self.device = device #torch.device(device)		
		self.window = window
		self.epochs = local_epochs
		# Default criterion set to MSE loss function
		self.criterion = nn.MSELoss().to(self.device)

	#-----------------------------------------
	def train_test_data(self, g_id, split_ratio):
		"""
		Returns train and test dataset for a given group.
		"""
		#dataframe = pd.read_csv(self.data_path+str(g_id)+"_val"+".csv")
		dataframe = pd.read_csv(self.data_path+str(g_id)+".csv")
	
		#all_data = dataframe['group_values'].values
		all_data = dataframe['load'].values
		all_data = all_data.astype('float32')
	    
		if self.normalize:
			#perform min/max scaling on the dataset which normalizes the data within a certain range of minimum and maximum values. 
			scaler = MinMaxScaler(feature_range=(0, 1))
			all_data_normalized = scaler.fit_transform(all_data.reshape(-1, 1))
			#print(all_data_normalized)
			all_data = all_data_normalized
			self.scaler = scaler

		# split into train and test sets (default: 80/20)
		if self.test_len is None:
			train_size = int(len(all_data) * split_ratio)
			test_size = len(all_data) - train_size
		else:
			train_size = len(all_data) - self.test_len		
			test_size = self.test_len
		
		train_data, test_data = all_data[:train_size], all_data[train_size:len(all_data)]
		if not self.normalize:
			train_data = train_data.reshape(-1, 1)
			test_data = test_data.reshape(-1, 1)
	
		return train_data, test_data

	#------------------------------------------
	def update_weights(self, model, global_round):
		"""
		Performs local update on the train data.
		"""
		x_train, y_train = utils.create_inout_sequences(self.train, self.window)
		train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
		train_loader = data_utils.DataLoader(train, batch_size=self.batch_size, shuffle=False)

		# Set optimizer for the local updates
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
		# Set mode to train model
		model.train()
		epoch_loss = []
		pre_w = model.state_dict()

		for i in range(self.epochs):
			batch_loss = 0.
			for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
				#print (x_batch.shape)
				seq, labels = x_batch.to(self.device), y_batch.to(self.device)
				labels = labels.view(-1, 1)
				optimizer.zero_grad()
				model.hidden_cell = model.init_hidden(batch_size=self.batch_size, device=self.device)
				y_pred = model(seq)
			
				loss = self.criterion(y_pred, labels)
				loss.backward()
				optimizer.step()

				batch_loss += loss.item()

			avg_loss = batch_loss / (batch_idx+1)
			if i%1 == 0:
				print(f'Global Round: {global_round:2}  Local epoch: {i+1:3}  Avg Loss: {avg_loss:10.8f}')
			epoch_loss.append(avg_loss)

		new_w = model.state_dict()
		rand_w = random_mask_weights(new_w, self.device, per_zeros=self.per_zeros)
             
		return rand_w, sum(epoch_loss) / len(epoch_loss)

	#----------------------------------------
	def inference(self, model):
		""" 
		Returns the inference loss on test data.
		"""
		x_test, y_test = utils.create_inout_sequences(self.test, self.window)
		test = data_utils.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
		test_loader = data_utils.DataLoader(test, batch_size=self.batch_size, shuffle=False)

		model.to(self.device)
				
		#print(next(model.parameters()).is_cuda)
		losses = []
		predicted_values = []
		actual_values = []

		model.eval()
		with torch.no_grad():
			for batch_idx, (seq, targets) in enumerate(test_loader):
				seq, targets = seq.to(self.device), targets.to(self.device)		
				targets = targets.view(-1, 1)
				#print(targets.shape)
			
				model.hidden_cell = model.init_hidden(self.device)
				outputs = model(seq)
				#print(outputs.shape)
			
				loss = self.criterion(outputs, targets)
				losses.append(loss)

				for out_tensor in outputs:
					predicted_values.append(out_tensor.item())
				#print(predicted_values[:5])
				for target_tensor in targets:
					actual_values.append(target_tensor.item())
				#print(actual_values[:5])				


		if self.normalize and self.scaler is not None:
			actual_values = self.scaler.inverse_transform(np.array(actual_values).reshape(-1, 1))
			predicted_values = self.scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))
		
			predicted_values = predicted_values[:,0]
			actual_values = actual_values[:,0]      

		
		return actual_values, predicted_values, losses

	#----------------------PREDICT AND UPDATE---------------------------
	def infer_and_update_weights(self, test_index, model, global_round):
		x_test, y_test = utils.create_inout_sequences(self.test, self.window)
		test = data_utils.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
		test_loader = data_utils.DataLoader(test, batch_size=1, shuffle=False)

		model.to(self.device)
				
		# Set optimizer for the local updates
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
		# Set mode to train model
		model.train()
		
		epoch_loss = []
		pre_w = model.state_dict()

		for i in range(self.test_epochs):
			epoch_loss = []
			predicted_values = []
			actual_values = []
			batch_loss = 0.
			
			for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
				#print (x_batch.shape)
				if batch_idx < test_index:
					continue
				seq, labels = x_batch.to(self.device), y_batch.to(self.device)
				#print (seq.shape)
				labels = labels.view(-1, 1)

				optimizer.zero_grad()
				model.hidden_cell = model.init_hidden(batch_size=1, device=self.device)
				y_pred = model(seq)

				for out_tensor in y_pred:
					predicted_values.append(out_tensor.item())
				#print(predicted_values[:5])
				for target_tensor in labels:
					actual_values.append(target_tensor.item())

				loss = self.criterion(y_pred, labels)
				loss.backward()
				optimizer.step()
				batch_loss += loss.item()

				if batch_idx > (test_index+self.test_range):
					break
				
			
			avg_loss = batch_loss / (batch_idx+1)
		
			epoch_loss.append(avg_loss)

			print(f'Global Round: {global_round:2}  Local epoch: {i+1:3}  Loss: {avg_loss:10.6f}')

		if self.normalize and self.scaler is not None:
			actual_values = self.scaler.inverse_transform(np.array(actual_values).reshape(-1, 1))
			predicted_values = self.scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))
		
			predicted_values = predicted_values[:,0]
			actual_values = actual_values[:,0]      

		new_w = model.state_dict()
		rand_w = random_mask_weights(new_w, self.device, self.per_zeros) 
             
		return rand_w, sum(epoch_loss) / len(epoch_loss), actual_values, predicted_values


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def random_mask_weights(w, device, per_zeros=0.5):
	rand_w = copy.deepcopy(w)

	#https://stackoverflow.com/questions/19597473/binary-random-array-with-a-specific-proportion-of-ones
	for key, value in w.items():
		#print(key, " shape: ", value.shape)
		#print(key, " : ",value)
		num_dim = len(list(value.shape))
		if num_dim < 2:
			total_elements = value.shape[0]
		else:	
			total_elements = value.shape[0] * value.shape[1]
		#print(total_elements)
		num_zeros = int(total_elements * per_zeros)
		num_ones = total_elements - num_zeros
		#print("zeros:", num_zeros, " ones:", num_ones)
		mask_arr = np.array([0] * num_zeros + [1] * num_ones)
		np.random.shuffle(mask_arr)
		if num_dim < 2:
			random_mask_arr = mask_arr.reshape(value.shape[0])
		else:
			random_mask_arr = mask_arr.reshape((value.shape[0], value.shape[1]))
		random_mask_tensor = torch.from_numpy(random_mask_arr).float().to(device)
		#random_mask_tensor = torch.from_numpy(random_mask_arr).float().to(torch.device(device))
		#print(random_mask_tensor)
		new_weight_value = value.mul(random_mask_tensor)
		
		rand_w[key] = new_weight_value
		#print(key, " new value: ", w[key])

	return rand_w
