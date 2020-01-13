import csv
import os
import sys
import io
import json
import mmap
from os import listdir
from os import path

#-----------------------------------
def make_dir(path, dir_name):
	try:
		out_path = path +"/"+dir_name
		if not os.path.exists(out_path):
			os.makedirs(out_path)
			return True
		else:
			print("Directory %s already exists"%dir_name)
			return False	
	except OSError:
		print ('Error: Creating directory. ' + out_path)
		sys.exit(0)

#------------------------------------
def save_json(filename, data, mode='w'):
	with open('{0}.json'.format(filename),mode) as f:
		json.dump(data, f)

#-----------------------------------
def load_json(filename):
    with io.open('{0}.json'.format(filename), encoding='utf-8') as f:
        return json.loads(f.read())
 
#-------------------------------------------
#Write text files with data
def write_txt(file_name, txt_data):
	with open(file_name+'.txt', 'w') as txtfile:
		for data in txt_data:
			txtfile.write(str(data))
			txtfile.write("\n")

#---------------------------------------
def read_txt(file_name):
	with open(file_name + ".txt", 'r') as txtfile:
		data = [ line.replace('\n', '') for line in txtfile.readlines()]
		
	return data

#---------------------------------------
#Write csv files with data using DictWriter
def write_csv(file_name, csv_data, fieldnames):
	#print ("Writing %s file ..."%file_name)
	with open(file_name+'.csv', 'w') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #writes the field names as headers
		writer.writeheader()
		for row in csv_data:
			writer.writerow(row)

#------------------------------------
#Append csv files with data using DictWriter
def append_csv(file_name, csv_data, fieldnames):
	print ("Appending %s file ..."%file_name)
	with open(file_name+'.csv', 'a') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		for row in csv_data:
			writer.writerow(row)

#---------------------------------------------
#Read csv files using DictReader
def read_csv(file_name, filter_field=None):
	csv_data = []
	#print ("Reading %s ...."%file_name)
	with open(file_name+'.csv', 'r') as csvfile:
		csvreader = csv.DictReader(csvfile)
		#get fieldnames from DictReader object and store in list
		fieldnames = csvreader.fieldnames 
		
		for row in csvreader:
			if filter_field is None:
				record = {}
				for field in fieldnames:
					record[field] = row[field]
			else:
				record = row[filter_field]
			
			csv_data.append(record)

	#print csv_data
	return fieldnames, csv_data 
#---------------------------------------------
#https://medium.com/anthony-fox/parsing-large-csv-files-with-python-854ab8f398ad
def get_csv_data(file_name):
	with open(file_name+".csv", "r") as csv_records:
		for record in csv.reader(csv_records):
			yield record

#-----------------------------------------------
#https://blog.nelsonliu.me/2016/07/30/progress-bars-for-python-file-reading-with-tqdm/
def get_num_lines(file_name):
    fp = open(file_name+".csv", "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    fp.close()
    return lines

#---------------------------------------
#Get all file names with a particular extension in a specified directory
def find_filenames_ext( path_to_dir, extension=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( extension ) ]

#--------------------------------------
#check whether a file exists
def doesFileExist(file_path, file_name):
    return path.exists(path.join(file_path, file_name))	

#remove contents of a directory
def empty_dir(dir_name):
	print("Removing all existing files in directory %s"%dir_name)
	filelist = [ f for f in os.listdir(dir_name)]
	for f in filelist:
		os.remove(os.path.join(dir_name, f))

#----------------------------------------
def merge_csv_files(merged_file_name):
	
	file_list = find_filenames_ext("./subset")
	
	f = open(merged_file_name, 'ab')
	master_csv = csv.writer(f)
	# Take the header of the first CSV and make it the master header	
	first_csv = open("./subset/"+file_list[0], 'rb')
	headers = first_csv.readline().strip().split(",")
	master_csv.writerow(headers)
	# Write remaining rows
	for line in first_csv:
		master_csv.writerow(line.strip().split(","))


	for counter, file_name in enumerate(file_list):
		if counter == 0:
			continue #skip the first file	
		# Read remaining CSVs and skip the first row
		with open("./subset/"+file_name, 'rb') as current_csv:
			for line_num, line in enumerate(current_csv):
				if line_num > 0:
					master_csv.writerow(line.strip().split(","))
		if (counter % 100 == 0):
			print ("Out of %d files, %d have been processed"%(len(file_list),counter))            

	f.close()
#-----------------------------------------------


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#if __name__ == "__main__":
 	
 	#print read_csv("AZ_postal_codes", ["Zip Code", "Place Name", "State", "State Abbreviation", "County", "Latitude", "Longitude"])[1]['Zip Code']
