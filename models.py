#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

#-------------------------LSTM Implementation with Batch Training support------------------------
#Helper Source: https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
#https://github.com/jessicayung/blog-code-snippets/blob/master/lstm-pytorch/lstm-baseline.py
#https://www.jessicayung.com/lstms-for-time-series-in-pytorch/
#https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py

import torch
from torch import nn

class LSTM(nn.Module):
    """
	input_size: 
		The number of expected features in the input. e.g., in 'airline-passengers' dataset for each time step (month) we have only 1 value i.e. total number of passengers, therefore the input size will be 1.
	hidden_state_size: 
		The number of features in the hidden state or the number of neurons (LSTM cells) in each layer

	num_layers:
		The number of recurrent layers. e.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1

	batch_first: boolean
		If True, then the input and output tensors are provided as (batch, seq, feature). Default: False

	output_size: 
		The number of items in the output, e.g., in 'airline-passengers' dataset, we want to predict the number of passengers for 1 month in the future, so the output size will be 1
    """

    def __init__(self, input_size=1, hidden_state_size=100, num_layers=1, batch_first=False, batch_size=1, output_size=1):
        super().__init__()

        self.input_size = input_size

        self.hidden_state_size = hidden_state_size

        self.num_layers = num_layers

        self.batch_size = batch_size        

        self.lstm = nn.LSTM(input_size, hidden_state_size, num_layers=self.num_layers, batch_first=batch_first)

        self.linear = nn.Linear(self.hidden_state_size, output_size)

    #Initialize the hidden state and the cell state to zero
    def init_hidden(self, batch_size, device):
        
        self.batch_size = batch_size
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_state_size, device=device), torch.zeros(self.num_layers, self.batch_size, self.hidden_state_size, device=device))

    # Forward pass through LSTM layer
    def forward(self, input_seq):
        
        self.lstm.flatten_parameters()

        #shape of lstm_out: [input_size, batch_size, hidden_state_size]
        #shape of self.hidden_cell: (a, b), where a and b denotes hidden and cell state respectively and both have shape (num_layers, batch_size, hidden_state_size).
        
        lstm_in = input_seq.view(self.batch_size, -1, self.input_size) # in: (batch_size, seq_len, input_size) [help: https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch]
        #print(lstm_in.shape)
        lstm_out, self.hidden_cell = self.lstm(lstm_in, self.hidden_cell)  # out: (batch_size, seq_len, hidden_size)
        #print(lstm_out)
        #print(lstm_out.shape)
        #print(lstm_out[:,-1, :].shape) #alternative: print(lstm_out[:,-1].shape)
        
        #Only take the output from the final timestep [Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction]
        #predictions = self.linear(lstm_out[:, -1]) #shape of in_to_linear: (batch_size, hidden_size) 
        predictions = self.linear(lstm_out[:, -1, :]) #shape of in_to_linear: (batch_size, hidden_size) 
        
        return predictions # shape of prediction: (batch_size, output_size)

