import matplotlib.pyplot as plt
import csv

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
def plot_predictions(fig_path, gid, test_values, test_predictions):
	plt.title('Timestep vs Power Consumption')
	plt.ylabel('Power Consumption')
	plt.grid(True)
	plt.autoscale(axis='x', tight=True)
	plt.plot(test_values)
	plt.plot(test_predictions)
	test_actual, = plt.plot(test_values, label='Actual test values')
	test_pred,  = plt.plot(test_predictions, label='Predicted test values')
	plt.legend(handles=[test_actual, test_pred])

	#plt.show()
	plt.savefig(fig_path+"/g"+str(gid)+"_actual_vs_pred.pdf", bbox_inches = "tight")
	plt.close()

#----------------------------------------
def plot_errors(fig_path, gid, errors, eid="single"):
	plt.title('Timestep vs Squared Errors')
	plt.ylabel('Squared Errors')
	plt.xlabel('Timesteps')
	plt.grid(True)
	plt.autoscale(axis='x', tight=True)
	plt.plot(range(len(errors)), errors)
	
	#plt.show()
	plt.savefig(fig_path+"/"+str(gid)+"_test_errors_"+eid+".pdf", bbox_inches = "tight")
	plt.close()

#------------------------------------------
def plot_losses(fig_path, gid, losses, lid="single"):
	plt.figure()
	plt.title('Testing Loss vs Timesteps')
	plt.grid(True)
	plt.plot(range(len(losses)), losses, color='r')
	plt.ylabel('MSE')
	plt.xlabel('Timesteps')
	plt.savefig(fig_path +"/"+ str(gid)+"_test_loss_"+ lid + ".pdf", bbox_inches = "tight")
	plt.close()

#---------------------------------------
def write_predictions(file_id, act_values, pred_values):
	with open(file_id+'.csv', 'w') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(['true_values', 'predicted_values'])
		for indx in range(len(act_values)):
			csv_writer.writerow([ act_values[indx], pred_values[indx] ])

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
