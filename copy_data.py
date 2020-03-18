import shutil
import helper

path = "./data"

meters = helper.read_txt(path+"/group_ids/g2")

for meter in meters:
	newPath = shutil.copy(path+"/meter_load_half_hr/" + meter + ".csv", path+"/sub_meter_load_half_hr/")
