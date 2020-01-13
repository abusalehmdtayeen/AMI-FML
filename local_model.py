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
from sklearn.preprocessing import MinMaxScaler
from torch import nn

#=============================================
class LocalModel(object):

	def __init__(self, gid, split_ratio=0.8, normalize=True, window = 24,local_epochs=10, device="cpu"):
		self.data_path = os.getcwd() + "/data/group_load/"
		self.normalize = normalize
		self.scaler = None
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
		dataframe = pd.read_csv(self.data_path+str(g_id)+"_sum"+".csv")
	
		all_data = dataframe['group_sum'].values
		all_data = all_data.astype('float32')
	    
		if self.normalize:
			#perform min/max scaling on the dataset which normalizes the data within a certain range of minimum and maximum values. 
			scaler = MinMaxScaler(feature_range=(0, 1))
			all_data_normalized = scaler.fit_transform(all_data.reshape(-1, 1))
			#print(all_data_normalized)
			all_data = all_data_normalized
			self.scaler = scaler

		# split into train and test sets (default: 80/20)
		train_size = int(len(all_data) * split_ratio)
		test_size = len(all_data) - train_size
		
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

			print(f'Global Round: {global_round:2}  Local epoch: {i:3}  Loss: {ep_loss:10.6f}')
             
		return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

	#----------------------------------------
	def inference(self, model):
		""" 
		Returns the inference loss.
		"""

		test_data = torch.FloatTensor(self.test).view(-1)	
		test_seq = create_inout_sequences(test_data, self.window)

		model.eval()
		loss = 0.0
		total_seq = 0
		test_predictions = []
		actual_predictions = []
		for seq, labels in test_seq:	
			seq, labels = seq.to(self.device), labels.to(self.device)
			with torch.no_grad():
				model.hidden = (torch.zeros(1, 1, model.hidden_layer_size, device=self.device), torch.zeros(1, 1, model.hidden_layer_size, device=self.device))
				# Inference
				outputs = model(seq)
				batch_loss = self.criterion(outputs, labels)
				loss += batch_loss.item()
				# Prediction
				test_predictions.append(model(seq).item())
				actual_predictions.append(labels.item())
			total_seq += 1		

		if self.normalized and self.scaler is not None:
			actual_predictions = self.scaler.inverse_transform(np.array(actual_predictions).reshape(-1, 1))
			test_predictions = self.scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
		
			test_predictions = test_predictions[:,0]
			actual_predictions = actual_predictions[:,0]      

		mse = math.sqrt(mean_squared_error(actual_predictions, test_predictions))
		
		return mse, loss

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
def test_inference(model, test_data, device):
	""" 
	Returns the loss on test data.
	"""

	model.eval()
	loss = 0.0

	criterion = nn.NLLLoss().to(device)
	

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
'''
