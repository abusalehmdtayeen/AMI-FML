#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
#Helper Source: https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
#https://github.com/jessicayung/blog-code-snippets/blob/master/lstm-pytorch/lstm-baseline.py

import torch
from torch import nn

class LSTM(nn.Module):
	"""
	input_size: Corresponds to the number of features in the input. For each time step we have only 1 value i.e. power consumption, therefore the input size will be 1.
	hidden_layer_size: Specifies the number of hidden layers along with the number of neurons in each layer. 
	output_size: The number of items in the output, since we want to predict the power consumption for 1 timestep in the future, the output size will be 1.
	"""
	def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
		super().__init__()
		self.hidden_layer_size = hidden_layer_size

		self.lstm = nn.LSTM(input_size, hidden_layer_size)

		self.linear = nn.Linear(hidden_layer_size, output_size)

		self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size), torch.zeros(1,1,self.hidden_layer_size))

	
	def forward(self, input_seq):
		self.lstm.flatten_parameters()
		lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
		predictions = self.linear(lstm_out.view(len(input_seq), -1))
        
		return predictions[-1]


