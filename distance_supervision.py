from mylib import find_author,langconv
import csv
import threading

import urllib
import http
import time
import requests

import os
import shutil




class spider_thread(threading.Thread):
	def __init__(self,row,writer,count=1):
		threading.Thread.__init__(self)
		self.row    = row
		self.writer = writer
		self.count = count

	def run(self):
		if self.count > 10:
			print ('超过错误次数，线程退出：',self.row[:2])
			return 
		book,person = self.row[:2]
		try:
			confidence = find_author(book,person)
			lst = [confidence]
			lst.extend(self.row)
			self.writer.writerow(lst)
		except (#urllib.error.HTTPError,
			    #urllib.error.URLError,
			    requests.exceptions.ConnectionError,
			    http.client.IncompleteRead,
			    ConnectionAbortedError,
			    ConnectionResetError) as e:
			# print ('报错',book,e)
			print ("等待2s,重启线程")
			time.sleep(2)
			spider_thread(self.row,self.writer,self.count+1).start()
		

def prepare_file():
	base_dir = 'data/dne/'
	if not os.path.isdir(base_dir+'book_person/'):
		os.mkdir(new_dir)
	target_file = base_dir+'book_person/book_person.csv'
	source_file = base_dir+'by_type/(\'Nh\', \'BOOK\').csv'
	if not os.path.isfile(target_file):
		shutil.copy(source_file,target_file) 


def distance_supervision():
	csv_file = 'data/dne/book_person/book_person.csv'
	with open(csv_file,newline="",encoding='utf8') as file_input:
		with open('data/dne/book_person/book_person_ds.csv','w+',newline='',encoding='utf-8') as file_output:
			reader = csv.reader(file_input)
			writer=csv.writer(file_output)
			for num,row in enumerate(reader):
				if not row:
					continue
				while threading.activeCount()>250:
					print ("线程过多，等待3秒；到了第%d行"%num)
					time.sleep(3)
				s_thread = spider_thread(row,writer)
				s_thread.start()
			while threading.activeCount()>1:
				print ("线程未全部结束，等待 1 秒；还有 %d 个线程"%(threading.activeCount()-1))
				time.sleep(1)


if __name__ == "__main__":
	prepare_file()
	distance_supervision()
