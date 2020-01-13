import matplotlib.pyplot as plt

#----------------------------------------
def plot_dataset(dataframe):
	plt.title('Timestep vs Power Consumption')
	plt.ylabel('Power Consumption')
	plt.xlabel('Timesteps')
	plt.grid(True)
	plt.autoscale(axis='x',tight=True)
	plt.plot(dataframe['group_sum'])
	plt.show()

#--------------------------------------
def create_inout_sequences(input_data, tw):
	"""
	Create sequences and corresponding labels for training
	Parameters
    ----------
	t_w : int, mandatory
		The size of window (the number of time steps can be used to make the prediction for the next time step) 
	Returns
    -------
    list: a list of tuples
	"""
	inout_seq = []
	L = len(input_data)
	for i in range(L-tw):
		train_seq = input_data[i:i+tw]
		train_label = input_data[i+tw:i+tw+1]
		inout_seq.append((train_seq ,train_label))
	
	return inout_seq
