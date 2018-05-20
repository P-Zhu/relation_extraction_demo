from mylib import find_author,langconv,baidu_result_num
import csv
import math

import threading
import urllib
import http
import time
import requests

import os
import shutil




class spider_thread(threading.Thread):
	"""find_author"""
	def __init__(self,row,writer,count=1):
		threading.Thread.__init__(self)
		self.row    = row
		self.writer = writer
		self.count = count

	def run(self):
		if self.count > 10:
			print ('超过错误次数，线程退出：',self.row[:2])
			return 
		person,book = self.row[:2]
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
		
class spider_thread2(threading.Thread):
	"""百度搜索结果数目"""
	def __init__(self,row,writer,count=1):
		threading.Thread.__init__(self)
		self.row    = row
		self.writer = writer
		self.count = count

	def run(self):
		if self.count > 10:
			print ('超过错误次数，线程退出：',self.row[:2])
			return 
		person,book = self.row[1:3]
		try:
			person_result_num = baidu_result_num(person)
			book_result_num = baidu_result_num(book)
			author_result_num = baidu_result_num("%s %s 作者"%(person,book))
			self.row.extend((math.log(person_result_num+1,10),
				math.log(book_result_num+1,10),
				math.log(author_result_num+1,10),
				))
			print(self.row,author_result_num,'%s %s 作者'%(person,book))
			self.writer.writerow(self.row)
		except (#urllib.error.HTTPError,
			    #urllib.error.URLError,
			    requests.exceptions.ConnectionError,
			    http.client.IncompleteRead,
			    ConnectionAbortedError,
			    ConnectionResetError) as e:
			# print ('报错',book,e)
			print ("等待2s,重启线程")
			time.sleep(2)
			spider_thread2(self.row,self.writer,self.count+1).start()


def prepare_file():
	base_dir = 'data/dne/'
	if not os.path.isdir(base_dir+'book_person/'):
		os.mkdir(new_dir)
	target_file = base_dir+'book_person/book_person.csv'
	source_file = base_dir+'by_type/(\'Nh\', \'BOOK\').csv'
	if not os.path.isfile(target_file):
		shutil.copy(source_file,target_file) 


def distance_supervision():
	csv_file1 = 'data/dne/book_person/book_person.csv'
	csv_file2 = 'data/dne/book_person/book_person_ds.csv'
	with open(csv_file1,newline="",encoding='utf8') as file_input:
		with open(csv_file2,'w+',newline='',encoding='utf-8') as file_output:
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

def adding_baidu_result_num():
	csv_file1 = 'data/dne/book_person/book_person_ds.csv'
	csv_file2 = 'data/dne/book_person/book_person_ds2.csv'
	with open(csv_file1,newline="",encoding='utf8') as file_input:
		with open(csv_file2,'w+',newline='',encoding='utf-8') as file_output:
			reader = csv.reader(file_input)
			writer=csv.writer(file_output)
			for num,row in enumerate(reader):
				if not row:
					continue
				while threading.activeCount()>250:
					print ("线程过多，等待3秒；到了第%d行"%num)
					time.sleep(3)
				s_thread = spider_thread2(row,writer)
				s_thread.start()
			while threading.activeCount()>1:
				print ("线程未全部结束，等待 1 秒；还有 %d 个线程"%(threading.activeCount()-1))
				time.sleep(1)


if __name__ == "__main__":
	prepare_file()
	# distance_supervision()
	# adding_baidu_result_num()
