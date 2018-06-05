from mylib import mySKF,TSVM
import numpy as np
import random
import csv
import os
import shutil
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.svm import SVC,NuSVC,LinearSVC 
from sklearn.metrics import recall_score,precision_score,accuracy_score
from sklearn.semi_supervised import LabelPropagation,LabelSpreading


class DNE_dataset:
	def __init__(self):
		self.csv_file = 'data/dne/dataset/book_person.csv'
		self.dataset_file = 'data/dne/dataset/dataset.csv'

	def prepare(self,source_file,n=-1):
		with open(source_file,'r',newline="",encoding='utf-8') as file_in:
			with open(self.csv_file,'w',newline="",encoding='utf8') as file_out:
				reader = csv.reader(file_in)
				writer = csv.writer(file_out)
				for num,row in enumerate(reader):
					# if n and num>n:
					# 	break
					confidence = float(row[0])
					if confidence < 0:
						_y = -1
					elif confidence == 0:
						_y = 0
					else:
						_y = 1
					row[0]=_y
					writer.writerow(row)

	def build(self,window=2):
		from mylib import MyWord2Vec,myLTP
		from mylib import Vectorization as vect
		mw2v = MyWord2Vec('data/w2v/',size=128)
		mw2v.load()
		myltp = myLTP(r'../ltp-model','mylib/pattern.txt')
		myltp.load([0,1,0,1,0])

		X = []
		y = []
		IR = []
		with open(self.csv_file,'r',newline="",encoding='utf-8') as file_in:
			reader = csv.reader(file_in)
			for row in reader:
				if not row:
					continue
				feature,label = self._vectorization(row,mw2v,myltp,vect,window)
				X.append(feature)
				y.append(label)

		self.X,self.y = np.array(X).astype(np.float),np.array(y).astype(np.int32)
		myltp.release()

	def _vectorization(self,row,w2v,ltp,vect,window):
		label,_,_,words,loc1,loc2,_,se1,se2,se3 = row
		label = float(label)
		loc1 = eval(loc1)
		loc2 = eval(loc2)
		words = eval(words)
		feature = vect(loc1,loc2,words,se1,se2,se3,w2v,ltp,window=window).vec()
		return feature,label

	def _random_shuffle(self,X,y):
		Xy = list(zip(X,y))
		random.shuffle(Xy)
		X,y = zip(*Xy)
		return np.array(X).astype(np.float),np.array(y).astype(np.int32)
		
	def save(self):
		with open(self.dataset_file,'w',newline="",encoding='utf-8') as file_out:
			writer = csv.writer(file_out)
			for feature,label in zip(self.X,self.y):
				lst = [label]
				lst.extend(feature)
				# lst.append(is_recorrect)
				writer.writerow(lst)

	def load(self,zero_included=False,random_shuffle=True):
		X = []
		y = []
		with open(self.dataset_file,'r',newline="",encoding='utf-8') as file_in:
			reader = csv.reader(file_in)
			for line in reader:
				label = int(line[0])
				feature = [float(i) for i in line[1:]]
				if label==0 and not zero_included:
					continue
				X.append(feature)
				y.append(label)
		if random_shuffle:
			self.X,self.y = self._random_shuffle(X,y)
		else:
			self.X,self.y = np.array(X).astype(np.float),np.array(y).astype(np.int32)

	def load_value(self,value,num_lim=10000):
		X = []
		y = []
		num = 0
		with open(self.dataset_file,'r',newline="",encoding='utf-8') as file_in:
			reader = csv.reader(file_in)
			for line in reader:
				label = int(line[0])
				feature = [float(i) for i in line[1:]]
				if label==value:
					X.append(feature)
					y.append(label)
					num += 1
				if num>=num_lim:
					break
		return np.array(X),np.array(y)
			

	def sample(self,size):
		X,y = self._random_shuffle(self.X,self.y)
		return X[:size],y[:size]

	def _get_precision_and_recall(self,y_test,y_pred):
		"""不计入未标记样本"""
		TP = FP = FN = TN = 0
		for y1,y2 in zip(y_test,y_pred):
			if y1==0:
				continue
			if y1==y2==1:
				TP += 1
			elif y1==1 and y2==-1:
				FN += 1
			elif y1==-1 and y2==1:
				FP += 1
			else:
				TN += 1
		print (TP,FP,FN,TN)
		try:
			return TP/(TP+FP),TP/(TP+FN),(TP+TN)/(TP+FN+FP+TN)
		except:
			return 0,0,0

	def _get_score(self,clf,X_train,y_train,X_test,y_test):
		"""获取分类器在数据集上的准确率"""
	
		# print('开始训练')
		clf.fit(X_train,y_train)
		# print('开始预测')
		y_pred = clf.predict(X_test)
		prec,reca,accu = self._get_precision_and_recall(y_test,y_pred)
		return prec,reca,accu

	def get_score(self,clf,X=None,y=None,splits=5):
		"""结合交叉验证 获取分类器在数据集上的准确率"""
		
		if X is None:
			X = self.X
			y = self.y
		precision = 0
		recall = 0
		mskf = mySKF(X,y,splits)
		cnt = 0
		for X_train,y_train,X_test,y_test in mskf:
			prec,reca,accu = self._get_score(clf,X_train,y_train,X_test,y_test)
			cnt += 1
			print('Split:{}, precision:{:.4f}, recall:{:.4f}, accu:{:.4f}'.format(cnt, prec, reca, accu))
			recall += reca
			precision += prec
		p,r = precision/splits, recall/splits
		f = 2*p*r/(p+r)
		return p,r,f


class Active_Learning:
	def __init__(self,X,y,clf,IR=None):
		self.X = X
		self.y = y
		self.clf = clf
		if IR == None:
			self.IR = []
		else:
			self.IR_index = [i for i in range(y.shape[0]) if IR[i]==1]
		self.full_index = list(range(y.shape[0]))
		self.consistent_index = list(range(y.shape[0]))
		self.inconsistent_index = []

	def single_iter(self):
		self.clf.fit(self.X[self.consistent_index],self.y[self.consistent_index])
		y_pred = self.clf.predict(self.X)
		incon_index = np.nonzero(y_pred - self.y)[0]
		self.inconsistent_index = self._index_plus(self.inconsistent_index,incon_index)
		self.inconsistent_index = self._index_minus(self.inconsistent_index,self.IR_index)
		self.consistent_index   = self._index_minus(self.full_index,self.inconsistent_index)
		
	def mainloop(self,iter_num=3):
		for _ in range(iter_num):
			self.single_iter()
			print (len(self.inconsistent_index))

	def _index_minus(self,index1,index2):
		return list(set(index1) - set(index2))

	def _index_plus(self,index1,index2):
		return list(set(index1) | set(index2))


def active_learning(dataset,clf,csv_file1,csv_file2):
	"""在数据集上训练分类模型，并在数据集上进行预测，标出预测错误的样本，以便人工查看"""
	
	print ('加载数据')
	dataset.load(zero_included=False,random_shuffle=False)

	print ('得到不一致样本的索引')

	print (dataset.y.shape)
	AL = Active_Learning(dataset.X,dataset.y,clf,dataset.IR)
	AL.mainloop(5)
	
	print ('写文件')

	lines = []
	zeros = []
	with open(csv_file1,'r',newline="",encoding='utf-8') as file_in:
		reader = csv.reader(file_in)
		for line in reader:
			if line[0] == '0':
				zeros.append(line)
			else:
				lines.append(line)
	lines = np.array(lines)

	with open(csv_file1,'w',newline="",encoding='utf-8') as file_out:
		writer = csv.writer(file_out)
		for line in zeros:
			writer.writerow(line)
		for line in lines[AL.consistent_index]:
			writer.writerow(line)
			
	with open(csv_file2,'w',newline="",encoding='utf-8') as file_out:
		writer = csv.writer(file_out)
		for line in lines[AL.inconsistent_index]:
			writer.writerow(line)


def build(dataset):
	# source_file = 'data/dne/book_person/book_person_ds2.csv'
	# dataset.prepare(source_file)
	dataset.build(window=2)
	dataset.save()


from metric_learn import LMNN


class LP:
	def __init__(self,lmnn=False,max_iter=1000,lm_num=200): 
		# self.clf =  LabelPropagation(kernel='knn',max_iter=1000,n_jobs=10,n_neighbors=25)
		self.clf =  LabelSpreading(kernel='knn',n_neighbors=25, max_iter=max_iter, alpha=0.2, n_jobs=-1)
		self.lmnn = lmnn
		self.lm_num = lm_num
		if lmnn:
			self.ml = LMNN(use_pca=False,max_iter=2000)
	def fit(self,X,y):
		if self.lmnn:
			nonzero_index = np.nonzero(y)
			index=random.sample(list(nonzero_index[0]),self.lm_num)
			X_ = X[index]
			y_ = y[index]
			print ('ml fitting')
			self.ml.fit(X_,y_)
			print ('transform')
			X=self.ml.transform(X)
		print ('lp fitting')
		zero_index = np.nonzero(y==0)
		negetive_index = np.nonzero(y==-1)
		positive_index = np.nonzero(y==1)
		y[zero_index] = -1
		y[negetive_index] = 2
		print (zero_index[0].shape,negetive_index[0].shape,positive_index[0].shape)
		self.clf.fit(X,y)
	def predict(self,X):
		print ('lp predict')
		if self.lmnn:
			X=self.ml.transform(X)
		y_pred = self.clf.predict(X)
		negative_index = np.nonzero(y_pred==-1)
		two_index = np.nonzero(y_pred==2)
		y_pred[negative_index] = 0
		y_pred[two_index] = -1
		return y_pred

def load_data_for_semi_supervised(num_un_half=500):
	X,y = dataset.load_value(value=1,num_lim=500+num_un_half)
	X_1,y_1 = X[:500],y[:500]
	X_01 = X[500:]
	y_01 = np.zeros((num_un_half,))

	X,y = dataset.load_value(value=-1,num_lim=500+num_un_half)
	X_n1,y_n1 = X[:500,],y[:500]
	X_0n1 = X[500:]
	y_0n1 = np.zeros((num_un_half,))

	X = np.concatenate((X_1,X_n1,X_01,X_0n1)) 
	y = np.concatenate((y_1,y_n1,y_01,y_0n1))
	# # clf = LP(lmnn=True,max_iter=1000,lm_num=lm_num)
	return X,y

def train_test(dataset,num_un_half=1500,lm_num=200):

	dataset.load(zero_included=True)
	X,y = dataset.X,dataset.y
	# index = np.nonzero(y)
	# X = X[index]
	# y = y[index]
	print (X.shape)
	
	clf = TSVM('mylib/svm_light_windows64/',kernel=2,
				gamma=1/665,kernel_cache=1000,
				weight_for_pn=1.,C=2.,verbosity=0.5)
	
	p,r,f = dataset.get_score(clf,splits=5)
	print (p,r,f)



if __name__ == '__main__':
	dataset = DNE_dataset()
	# build(dataset)
	
	# active_learning(dataset,clf, 'data/dne/dataset/book_person.csv',   
		# 'data/dne/dataset/book_person_al.csv')
	
	train_test(dataset)