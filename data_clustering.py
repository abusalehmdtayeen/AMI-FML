import datetime 
import time
import sys
import os
import csv
import random

from operator import itemgetter
from tqdm import tqdm

import helper 

#=====================================
base_path = os.getcwd()
data_path = base_path + "/data/"
#====================================

#https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print ('%r  %2.2f s' %(method.__name__, (te - ts)))
        return result

    return timed

#------------------------------------------
def make_meter_groups(meter_ids, group_size = 100, group_limit=None):
	'''
	Make groups of meters.

	Parameters
    ----------
	group_size : int, optional
		The size of each meter group (default is 100)
	group_limit : int, optional
		The number of groups to be created from the meters (default is None)
	Returns
    -------
    dict
        a dict of meter groups
	'''
	meter_groups = {}
	random.seed(2)
	random.shuffle(meter_ids)
	group_num = 1
	for i in range((len(meter_ids) // group_size) + 1):
		group = meter_ids[i*group_size:i*group_size + group_size]
		if len(group) != group_size: #make sure all groups are same size
			continue 
		meter_groups[group_num] = group
		if group_limit is not None and group_limit == group_num:
			break
			
		group_num += 1

	print("Number of groups: %d"%len(meter_groups.keys()))
	#write the meter ids of each group in file
	dir_exists = helper.make_dir(data_path, "group_ids")
	if not dir_exists:
		helper.empty_dir(data_path + "group_ids")

	for gid in meter_groups.keys():
		helper.write_txt(data_path+"group_ids"+"/"+"group-"+str(gid), meter_groups[gid])

	#print (meter_groups)
	return meter_groups

#-----------------------------------------------------------------
@timeit
def create_group_meter_data(meter_groups, time_scale=1.0):
	'''
	Create half an hour/hourly data for each group of meters sorted by time

	Parameters
    ----------
	time_scale : float, optional
		granularity of data (default is 1.0 which means hourly data)
	'''

	if (time_scale >= 1.0):
		suffix = "_1hr"
		time_len = 24
	else:
		suffix = "_half_hr"
		time_len = 48

	#create output directory for group data 
	dir_exists = helper.make_dir(data_path, "group_load")
	if not dir_exists:
		helper.empty_dir(data_path + "group_load")

	for gid in meter_groups:
		group_meters = meter_groups[gid]
		meter_data = {}
		print ("Processing group %d"%gid)
		for meter_id in group_meters:			
			with open(data_path+"/meter_load"+suffix+"/"+str(meter_id)+".csv", 'r') as csvfile:
				csvreader = csv.reader(csvfile)
				try:
					next(csvreader) # This skips the first row (header) of the CSV file.
				except StopIteration:
					print("No rows")
					continue
				for row in csvreader:
					day = int(row[0])
					time = int(row[1])
					load = float(row[2])
					if meter_id not in meter_data:
						meter_data[meter_id] = {day: [{'time': time, 'load': load}]}
					else:
						if day not in meter_data[meter_id]:
							meter_data[meter_id][day] = [{'time': time, 'load': load}]
						else:
							meter_data[meter_id][day].append({'time': time, 'load': load})

		sorted_meter_data = {}
		meters = [ key for key in meter_data ]
		#print(meters)
		for meter in tqdm(meters):
			#print("Processing meter with ID: %s"%meter)
			sorted_meter_data[meter] = {}
			days = meter_data[meter].keys()
			days = sorted(days)
			#print(days)
			for day in days:
				per_day_data = []
				time_loads = meter_data[meter][day]
				time_loads = sorted(time_loads, key=itemgetter('time'))
				
				assert len(time_loads) == time_len, "Meter {} does not have data for a full day".format(meter)
				
				for t_indx in range(len(time_loads)):
					#print (time_loads[t_indx], time_loads[t_indx+1])
					load = time_loads[t_indx]['load']
					per_day_data.append(load)
 
				sorted_meter_data[meter][day] = per_day_data 
			 		
		total_time_steps = 0
		days = []
		#find the total number of time steps
		for m, dt in sorted_meter_data.items():
			for day, data in dt.items():		
				total_time_steps += len(data)
				days.append(day)
			break

		#print(total_time_steps)

		same_days = True
		#check same days
		for mt_indx in range(len(meters)-1):
			for m, dt in sorted_meter_data.items():
				if m != meters[mt_indx]:
					continue
				mt_days = [ day for day, data in dt.items() ]
				break
			
			#print(mt_days)
			for m, dt in sorted_meter_data.items():
				if m != meters[mt_indx+1]:
					continue
				mt_next_days = [ day for day, data in dt.items() ]
				break
			#print(mt_next_days)

			same_days = same_days and (sorted(mt_days) == sorted(mt_next_days)) 	
		if not same_days:
			print("Group %d does not have same day data"%gid)
			continue

		days = sorted(days)
		num_time_steps = time_len
		group_data_sum = []
		
		total_entries = 0
		for day in days:			
			for t_indx in range(num_time_steps):
				#print (day, t_indx, ":" )
				group_sum = 0
				for meter in meters:
					#print(sorted_meter_data[meter][day][t_indx])
					group_sum += sorted_meter_data[meter][day][t_indx]
				 
				group_data_sum.append({'day_time': str(day)+"-"+str(t_indx+1), 'group_sum':group_sum}) 
				total_entries += 1

		assert len(group_data_sum) == total_entries, "Data length does not match" 
		#print(len(group_data_sum))
		helper.write_csv(data_path+"group_load/"+"g"+str(gid)+"_sum", group_data_sum, ["day_time", "group_sum"])
	
#----------------------------------

#-----------------------------------
if __name__ == "__main__":
	
	suffix = "_half_hr"
	#get all meter ids from file names
	meter_files =  helper.find_filenames_ext(data_path+"/meter_load"+suffix+"/", ".csv")
	meters = [ meter_file[ : meter_file.rindex(".")] for meter_file in meter_files ]
	print("Total number of meter ids = %d"%len(meters))
	
	#make two groups with each with size 3
	meter_groups = make_meter_groups(meter_ids=meters, group_size=3, group_limit=2)
	#generate half hr data for each group
	create_group_meter_data(meter_groups, 0.5)

	
