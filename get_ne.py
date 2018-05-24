from mylib import myLTP
import csv
import os
from collections import defaultdict



def find_dne(txt_file,csv_file,ltp):
	"""找到文本中的实体对，并输出到csv文件中"""
	with open(txt_file,encoding='utf-8') as file_in:
		with open(csv_file,'w+',newline='',encoding='utf-8') as file_out:
			writer = csv.writer(file_out)
			for line in file_in:
				line = line.replace('　',' ')
				rs=ltp.get_dne(line)
				for (type1,start1,end1),(type2,start2,end2),words in rs:
					d = min(abs(start1 - end2),abs(start2 - end1))
					if d > 6: # 当实体距离大于6
						continue
					loc1 = (start1,end1)
					loc2 = (start2,end2)
					ne1 = ''.join(words[start1:end1])
					ne2 = ''.join(words[start2:end2])
					new_list=(type1,type2,loc1,loc2,ne1,ne2,words,txt_file.split("/")[-1])
					writer.writerow(new_list)


def find_dne_for_dir(txt_dir,csv_dir,ltp):
	"""对目录下的文本文件，进行 find_dne """
	for txt_file_name in os.listdir(txt_dir):
		txt_file = txt_dir + txt_file_name
		if not os.path.isfile(txt_file):
			continue
		csv_file = csv_dir+txt_file_name[:-3]+'csv'
		print (txt_file)
		find_dne(txt_file,csv_file,ltp)

 
def class_dne_by_type(csv_dir,new_dir):
	"""按照实体对的类型，写入不同的文件中"""
	if not os.path.isdir(new_dir):
		os.mkdir(new_dir)

	dne_by_type = defaultdict(list)
	for csv_file_name in os.listdir(csv_dir):
		csv_file = csv_dir + csv_file_name
		if not os.path.isfile(csv_file):
			continue
		with open(csv_file,newline="",encoding='utf8') as file_input:
			reader = csv.reader(file_input)
			for row in reader:
				if not row:
					continue
				type1,type2,loc1,loc2,ne1,ne2,words,source = row
				if type2 > type1:
					loc1,loc2 = loc2,loc1
					ne1, ne2  = ne1, ne2
				new_list = [ne1,ne2,words,loc1,loc2,source]
				dne_by_type[(type1,type2)].append(new_list)

	for type_,dnes in dne_by_type.items():
		with open(new_dir+str(type_)+'.csv','w',newline="",encoding='utf8') as file_output:
			writer = csv.writer(file_output)
			writer.writerows(dnes)


if __name__ == "__main__":
	myltp = myLTP(r'../ltp-model','mylib/pattern.txt')
	myltp.load([0,1,1,0,0])
	find_dne_for_dir('data/txt/','data/dne/',myltp)
	class_dne_by_type('data/dne/','data/dne/by_type/')
	myltp.release()

