#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
#Helper Source: https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/federated_main.py

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import helper
import utils
import torch

from local_model import LocalModel
from models import LSTM

#==========PARAMETERS=======================
global_epochs = 3
local_epochs = 2
frac = 0.7 #fraction of groups to choose
split_ratio = 0.8 #split ratio for train data
normalize_data = True
window_size = 48
#===========================================

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU.
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

#--------------------------------------------
if __name__ == '__main__':
	start_time = time.time()

    # load meter groups
	groups = helper.find_filenames_ext(base_path + "/data/group_load/")
	group_ids = [ group[ : group.rindex("_")] for group in groups] 
	num_groups = len(groups)

	# create model object
	global_model = LSTM()
    
    # Set the model to train and send it to device.
	global_model.to(device)
	global_model.train()
    #print(global_model)
    # copy weights
	global_weights = global_model.state_dict()

    # Training
	train_loss = []
    
	#~~~~~~~~~~~~~~~~~CHOOSE ALL GROUPS~~~~~~~~~~~~~~~
	local_models = []
	group_indices = np.arange(num_groups)
	for indx in group_indices:
		local_model = LocalModel(group_ids[indx], split_ratio, normalize=normalize_data, window=window_size, local_epochs=local_epochs, device=device)
		local_models.append(local_model)
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	for epoch in range(global_epochs):
		local_weights, local_losses = [], []
	
		global_model.train()
		#~~~~~~~~~~~~~~~~~CHOOSE FRACTION OF ALL GROUPS RANDOMLY~~~~~~~~~~~ 
		#m = max(int(frac * num_groups), 1)
		#group_indices = np.random.choice(range(num_groups), m, replace=False)
		#local_models = []
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		for indx in group_indices:
			print("Training group with ID: %s"%str(group_ids[indx]))
			#~~~~~~~~~~~~~~~when FRACTION OF ALL GROUPS are choosen randomly~~~~~~~~~~~~
			#local_model = LocalModel(group_ids[indx], split_ratio, normalize=normalize_data, window=window_size, local_epochs=local_epochs, device=device)
			#local_models.append(local_model)
			#~~~~~~~~~~~~~~~~~when ALL GROUPS are chosen~~~~~~~~~~~~~~~~~~~~~~~~~~~
			local_model = local_models[indx]
			#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
			local_weights.append(copy.deepcopy(w))
			local_losses.append(copy.deepcopy(loss))

        # update global weights
		global_weights = average_weights(local_weights)

        # update global weights
		global_model.load_state_dict(global_weights)

		loss_avg = sum(local_losses) / len(local_losses)
		train_loss.append(loss_avg)

		print('Global Training Round : {}, Average loss {:.3f}'.format(epoch, loss_avg))

	print('\n Total Training Time: {0:0.4f}'.format(time.time()-start_time))
	#------------------------------------------------------------------------
	#~~~~~~~~~~~~~~~when ALL GROUPS are chosen~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Test inference after completion of training
	print(f' \n Results after {global_epochs} global rounds of training:')	
	group_losses = []
    global_model.eval()
	
	for indx in group_indices:
		local_model = local_models[indx]
		rmse, losses = local_model.inference(model=global_model)
		group_losses.append(losses)

		print("Test score of group %s: %.2f RMSE"%(str(group_ids[indx]), rmse))
   
 
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
	
    #--------------------------------------------------------- 
	helper.make_dir(base_path, "figures")	
	print("Plotting federated test loss curves")
	for indx in group_indices:
		gid = group_ids[indx]
		g_losses = group_losses[indx]
		utils.plot_losses(base_path + "/figures/", gid, g_losses, lid="federated")
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

