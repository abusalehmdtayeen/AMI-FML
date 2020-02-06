#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import math
import torch
import numpy as np
import pandas as pd
import utils
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn

#=============================================
class LocalModel(object):

	def __init__(self, gid, split_ratio=0.8, test_len=None, test_range=1, normalize=True, window = 24,local_epochs=10, test_epochs=1, device="cpu"):
		#self.data_path = os.getcwd() + "/data/group_load/"
		self.data_path = os.getcwd() + "/data/meter_load_half_hr/"
		self.normalize = normalize
		self.scaler = None
		self.test_len = test_len
		self.test_range = test_range # the number of test points to predict before updating the model
		self.test_epochs = test_epochs
		self.train, self.test = self.train_test_data(gid, split_ratio)
		self.device = torch.device(device)		
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
		train_tensor = torch.FloatTensor(self.train).view(-1)
		train_seq = utils.create_inout_sequences(train_tensor, self.window)

		# Set optimizer for the local updates
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
		# Set mode to train model
		model.train()
		epoch_loss = []
		
		for i in range(self.epochs):
			batch_loss = []
			for seq, labels in train_seq:
				seq, labels = seq.to(self.device), labels.to(self.device)
				optimizer.zero_grad()
				model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size, device=self.device), torch.zeros(1, 1, model.hidden_layer_size, device=self.device))
				y_pred = model(seq)

				loss = self.criterion(y_pred, labels)
				loss.backward()
				optimizer.step()
				batch_loss.append(loss.item())

			ep_loss = sum(batch_loss)/len(batch_loss)
			epoch_loss.append(ep_loss)

			print(f'Global Round: {global_round:2}  Local epoch: {i+1:3}  Loss: {ep_loss:10.6f}')
             
		return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

	#----------------------------------------
	def inference(self, model):
		""" 
		Returns the inference loss on test data.
		"""
		model.to(self.device)
		test_tensor = torch.FloatTensor(self.test).view(-1)	
		test_seq = utils.create_inout_sequences(test_tensor, self.window)

		model.eval()
		#print(next(model.parameters()).is_cuda)
		losses = []
		total_seq = 0
		test_predictions = []
		actual_predictions = []
		for seq, labels in test_seq:	
			seq, labels = seq.to(self.device), labels.to(self.device)
			with torch.no_grad():
				model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size, device=self.device), torch.zeros(1, 1, model.hidden_layer_size, device=self.device))
				# Inference
				outputs = model(seq)
				batch_loss = self.criterion(outputs, labels)
				losses.append(batch_loss.item())
				
				test_predictions.append(outputs.item())
				actual_predictions.append(labels.item())
			total_seq += 1		

		if self.normalize and self.scaler is not None:
			actual_predictions = self.scaler.inverse_transform(np.array(actual_predictions).reshape(-1, 1))
			test_predictions = self.scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
		
			test_predictions = test_predictions[:,0]
			actual_predictions = actual_predictions[:,0]      

		#rmse = math.sqrt(mean_squared_error(actual_predictions, test_predictions))
		
		return actual_predictions, test_predictions, losses

	#----------------------PREDICT AND UPDATE---------------------------
	def infer_and_update_weights(self, test_index, model, global_round):
		model.to(self.device)
		test_tensor = torch.FloatTensor(self.test).view(-1)	
		test_seq = utils.create_inout_sequences(test_tensor, self.window)

		# Set optimizer for the local updates
		optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
		# Set mode to train model
		model.train()
		
		epoch_loss = []

		for i in range(self.test_epochs):
			batch_loss = []
			test_predictions = []
			actual_predictions = []
			for indx, (seq, labels) in enumerate(test_seq):
				if indx < test_index:
					continue
				seq, labels = seq.to(self.device), labels.to(self.device)
				optimizer.zero_grad()
				model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size, device=self.device), torch.zeros(1, 1, model.hidden_layer_size, device=self.device))
				y_pred = model(seq)

				test_predictions.append(y_pred.item())
				actual_predictions.append(labels.item())

				loss = self.criterion(y_pred, labels)
				loss.backward()
				optimizer.step()
				batch_loss.append(loss.item())
				if indx > (test_index+self.test_range):
					break
					
			ep_loss = sum(batch_loss)/len(batch_loss)
			epoch_loss.append(ep_loss)

			print(f'Global Round: {global_round:2}  Local epoch: {i+1:3}  Loss: {ep_loss:10.6f}')

		if self.normalize and self.scaler is not None:
			actual_predictions = self.scaler.inverse_transform(np.array(actual_predictions).reshape(-1, 1))
			test_predictions = self.scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
		
			test_predictions = test_predictions[:,0]
			actual_predictions = actual_predictions[:,0]      
             
		return model.state_dict(), sum(epoch_loss) / len(epoch_loss), actual_predictions, test_predictions


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
