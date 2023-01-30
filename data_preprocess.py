import datetime 
import time
import sys
import csv
import os
import pymongo
from pymongo import errors
from bson import json_util
from bson.json_util import dumps
from operator import itemgetter
from tqdm import tqdm

import helper 

#=====================================
base_path = os.getcwd()
raw_data_path = "/CER_Electricity/CER_Electricity_Revised_March_2012" #the raw zip files for meters
processed_data_path = "/data" #output folder for the processed data
#------------------------------------
dbName = "smart_meter"
collectionName = "load_profiles"

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
@timeit
def filter_raw_data(file_name, meter_ids):
	residential_data = []
	fields = ["MeterID", "DayTimeCode", "Electricity"]

	# open raw file
	with open(file_name, "rb") as fp:
		#iterate over file line-by-line
		num_lines = 0
		while True:
			line = fp.readline()
			if line: #valid line 
				#strip line of newline symbols and split line by spaces into list (of number strings)
				line_segments = line.strip().split(' ')
				#check whether meterID exists in the residential meterIDs
				if line_segments[0] in meter_ids:	
					#add to residential data	
					residential_data.append({"MeterID": line_segments[0], "DayTimeCode": line_segments[1], "Electricity": line_segments[2]})            	
			else:
				break #no line exists
			num_lines += 1

		print ("%d records extracted out of total %d raw records"%(len(residential_data), num_lines))
		#separate the file name from path 
		processed_file_name = file_name[file_name.rindex("/")+1:file_name.rindex(".")]
		
		helper.write_csv(base_path+processed_data_path+"/load/"+processed_file_name, residential_data, fields)

#--------------------------------------------
#connection to Mongo DB
def connect_mongodb(host='localhost', port=27017):
 	try:
 		conn=pymongo.MongoClient(host, port)
 		print "Connected successfully!!!"
 	except pymongo.errors.ConnectionFailure, e:
 		print "Could not connect to MongoDB: %s" % e 

 	return conn

#--------------------------------------
def setup_db(conn):
	db = conn[dbName] # access "smart_meter" database
	
	try:
		# create a new collection named "load_profiles" for the "smart_meter" database
		conn.db.create_collection(collectionName)	
	except errors.CollectionInvalid as err: #collection already exists
		pass
		#print ("PyMongo ERROR:", err, "\n")
		
#-----------------------------------
def insert_doc_db(conn, doc):
	db = conn[dbName]	
	collection = db[collectionName]
	result_object = collection.insert_one(doc)

	if result_object.acknowledged:
		#print("Document inserted successfully")
		return True
	else:
		#print("!!Document could not be inserted")
		return False

#---------------------------------------
def insert_docs_db(conn, docs):
	db = conn[dbName]	
	collection = db[collectionName]
	result_object = collection.insert_one(doc)

	result = collection.insert_many(docs)

#------------------------------------
@timeit
def create_db(conn):
	filtered_files =  helper.find_filenames_ext(base_path+processed_data_path+"/load/", ".csv")
	#print (filtered_files)
	setup_db(conn)
	
	for file_name in filtered_files:
		print ("Processing file %s"%file_name)
		file_name_wo_indx = file_name[:file_name.rindex(".")]
		num_file_records = helper.get_num_lines(base_path+processed_data_path+"/load/"+file_name_wo_indx)
		print ("Total file records: %d"%num_file_records)
		iter_records = iter(helper.get_csv_data(base_path+processed_data_path+"/load/"+file_name_wo_indx))
  		next(iter_records)  # Skipping the column names
		total_records = 0
		
		for row in iter_records:
			day = row[1][:3]
			time = row[1][3:]
			load = row[2]
			meter_doc = {'meter_id': row[0], 'day': int(day), 'time': int(time), 'load': float(load)}
			insert_status = insert_doc_db(conn, meter_doc)
			if (insert_status):
				total_records += 1
			if total_records % 100000 == 0:
				print ("%.2f percent completed"%((float(total_records)/num_file_records)*100))			

		print("Total %d records inserted out of %d file records"%(total_records, num_file_records))	
		assert total_records == (num_file_records-1), "Number of inserted records are not equal to number of records read"

#-----------------------------------
#create data for each meter sorted by day and time
@timeit
def create_per_meter_data(time_scale=0.5):
	if (time_scale < 1.0):
		suffix = "_half_hr"
		step = 1
	else:
		suffix = "_1hr"
		step = 2

	filtered_files =  helper.find_filenames_ext(base_path+processed_data_path+"/load/", ".csv")

	helper.make_dir(base_path+processed_data_path, "meter_load"+suffix)

	output_path = base_path+processed_data_path+"/meter_load"+suffix+"/"
	
	incomplete_data_records = []
	for file_name in filtered_files:
		print ("Processing file %s"%file_name)
		meter_data = {}
		with open(base_path+processed_data_path+"/load/"+file_name, 'r') as csvfile:
			csvreader = csv.reader(csvfile)
			try:
				next(csvreader) # This skips the first row (header) of the CSV file.
			except StopIteration:
				print("No rows")
				continue
			for row in csvreader:
				day = int(row[1][:3])
				time = int(row[1][3:])
				load = float(row[2])
				meter_id = row[0]
				if meter_id not in meter_data:
					meter_data[meter_id] = {day: [{'time': time, 'load': load}]}
				else:
					if day not in meter_data[meter_id]:
						meter_data[meter_id][day] = [{'time': time, 'load': load}]
					else:
						meter_data[meter_id][day].append({'time': time, 'load': load})

		meters = meter_data.keys()
		for meter in tqdm(meters):
			#print("Processing meter with ID: %s"%meter)
			per_meter_data = []
			days = meter_data[meter].keys()
			days = sorted(days)
			#print(days)
			for day in days:
				time_loads = meter_data[meter][day]
				time_loads = sorted(time_loads, key=itemgetter('time'))
				
				if len(time_loads) != (48/step):
					incomplete_data_records.append({'MeterID': meter, 'Day': day})
					continue
				else:
					new_time = 1
					for t_indx in range(0, len(time_loads), step):
						#print (time_loads[t_indx], time_loads[t_indx+1])
						if step == 2:
							new_load = time_loads[t_indx]['load'] + time_loads[t_indx+1]['load']
						elif step == 1:
							new_load = time_loads[t_indx]['load']
						per_meter_data.append({'day': day, 'time': new_time, 'load': new_load})  
						new_time += 1			 

			#write per meter data
			if not helper.doesFileExist(output_path, meter+".csv"):
				helper.write_csv(output_path+meter, per_meter_data, ['day', 'time', 'load'])
			else:
				print("File %s exists"%(meter+".csv"))
			
	helper.write_csv(base_path+processed_data_path+"/"+"incomplete"+suffix ,incomplete_data_records, ['MeterID', 'Day'])


#----------------------------------
@timeit
def check_incomplete_data(time_scale=0.5): #time_scale = 0.5 = half an hour data (default)
	if (time_scale < 1.0):
		suffix = "_half_hr"
		step = 1
	else:
		suffix = "_1hr"
		step = 2

	filtered_files =  helper.find_filenames_ext(base_path+processed_data_path+"/load/", ".csv")
	output_path = base_path+processed_data_path+"/meter_load"+suffix+"/"

	incomplete_data_records = {}
	with open(base_path+processed_data_path+"/"+"incomplete"+suffix+".csv", 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		try:
			next(csvreader) # This skips the first row (header) of the CSV file.
		except StopIteration:
			print("No rows")
			sys.exit(0)
		for row in csvreader:	
			if row[0] not in incomplete_data_records:
				incomplete_data_records[row[0]] = [int(row[1])]
			else:
				incomplete_data_records[row[0]].append(int(row[1]))

	incomplete_meters = incomplete_data_records.keys()
	print("%d meters have incomplete data"%(len(incomplete_meters)))
	#print(incomplete_data_records)
	
	final_incomplete_records = []
	
	meter_data = {}
	for file_name in filtered_files:
		print ("Processing file %s"%file_name)
		with open(base_path+processed_data_path+"/load/"+file_name, 'r') as csvfile:
			csvreader = csv.reader(csvfile)
			try:
				next(csvreader) # This skips the first row (header) of the CSV file.
			except StopIteration:
				print("No rows")
				continue
			for row in csvreader:
				day = int(row[1][:3])
				time = int(row[1][3:])
				load = float(row[2])
				meter_id = row[0]	
				
				if meter_id not in incomplete_meters:
					continue
				if meter_id in incomplete_meters and day not in incomplete_data_records[meter_id]:
					continue
	
				if meter_id not in meter_data:
					meter_data[meter_id] = {day: [{'time': time, 'load': load}]}
				else:
					if day not in meter_data[meter_id]:
						meter_data[meter_id][day] = [{'time': time, 'load': load}]
					else:
						meter_data[meter_id][day].append({'time': time, 'load': load})

	
		
	meters = meter_data.keys()
	for meter in meters:
		per_meter_data = []
		days = meter_data[meter]
		days = sorted(days)
		for day in days:
			time_loads = meter_data[meter][day]
			time_loads = sorted(time_loads, key=itemgetter('time'))
				
			if len(time_loads) != (48/step):
				final_incomplete_records.append({'MeterID': meter, 'Day': day})
				continue
			else:
				new_time = 1
				for t_indx in range(0,len(time_loads),step):
					#print (time_loads[t_indx], time_loads[t_indx+1])
					if step == 2:
						new_load = time_loads[t_indx]['load'] + time_loads[t_indx+1]['load']
					elif step == 1:
						new_load = time_loads[t_indx]['load']

					per_meter_data.append({'day': day, 'time': new_time, 'load': new_load})  
					new_time += 1			 

		#write per meter data
		if helper.doesFileExist(output_path, meter+".csv"):
			if len(per_meter_data) != 0 :
				helper.append_csv(output_path+meter, per_meter_data, ['day', 'time', 'load'])
			else:
				print(meter+" has no new data")
		else:
			print("File %s does not exist"%(meter+".csv"))
	
	helper.write_csv(base_path+processed_data_path+"/"+"final_incomplete"+suffix ,final_incomplete_records, ['MeterID', 'Day'])

#-----------------------------------
if __name__ == "__main__":

	make_dir(base_path+processed_data_path, "load")
	make_dir(base_path+processed_data_path, "meter")
	
	#get the residential smart meter ids
	meter_ids = helper.read_csv(base_path+processed_data_path+"/meter/"+"residential_sm_ids", ["ID", "Code"], filter_field="ID")
	#get all raw file names
	files =  helper.find_filenames_ext(base_path+raw_data_path, ".zip") #.txt
	#filter only residential meter data from raw files
	for file_name in files:
		file_path = base_path+raw_data_path+"/"
		if not helper.doesFileExist(file_path, file_name):
			print ("Filtering data from file: %s"%file_name)
			filter_raw_data(file_path+file_name, meter_ids)
		else:
			print("%s exists"%file_name)
	#----------------------------------------
	conn = connect_mongodb()	
	create_db(conn)
	create_per_meter_data(0.5)
	check_incomplete_data(0.5)
	
