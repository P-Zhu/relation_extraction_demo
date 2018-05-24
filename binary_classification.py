from mylib import MyWord2Vec,myLTP,Vectorization,mySKF,TSVM
import numpy as np
import random
import csv
import os
import shutil
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.svm import SVC,NuSVC,LinearSVC 
from sklearn.metrics import recall_score,precision_score



class DNE_dataset:
	def __init__(self):
		self.csv_file = 'data/dne/dataset/book_person.csv'
		self.dataset_file = 'data/dne/dataset/dataset.csv'

	def prepare(self,source_file):
		with open(source_file,'r',newline="",encoding='utf-8') as file_in:
			with open(self.csv_file,'w',newline="",encoding='utf8') as file_out:
				reader = csv.reader(file_in)
				writer = csv.writer(file_out)
				for row in reader:
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
		mw2v = MyWord2Vec('data/w2v/',size=128)
		mw2v.load()
		myltp = myLTP(r'../ltp-model','mylib/pattern.txt')
		myltp.load([0,1,0,1,0])

		X = []
		y = []
		with open(self.csv_file,'r',newline="",encoding='utf-8') as file_in:
			reader = csv.reader(file_in)
			for row in reader:
				feature,label = self._vectorization(row,mw2v,myltp,window)
				X.append(feature)
				y.append(label)

		self.X,self.y = self._random_shuffle(X,y)
		myltp.release()

	def _vectorization(self,row,w2v,ltp,window):
		label,_,_,words,loc1,loc2,_,se1,se2,se3 = row
		label = float(label)
		loc1 = eval(loc1)
		loc2 = eval(loc2)
		words = eval(words)
		feature = Vectorization(loc1,loc2,words,se1,se2,se3,w2v,ltp,window=window).vec()
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
				writer.writerow(lst)

	def load(self,zero_included=False):
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
		self.X,self.y = np.array(X),np.array(y)

	def sample(self,size):
		X,y = self._random_shuffle(self.X,self.y)
		return X[:size],y[:size]

	def _get_precision_and_recall(self,y_test,y_pred):
		"""不计入未标记样本"""
		TP = FP = FN =0
		for y1,y2 in zip(y_test,y_pred):
			if y1==y2==1:
				TP += 1
			elif y1==1 and y2==-1:
				FN += 1
			elif y1==-1 and y2==1:
				FP += 1
		return TP/(TP+FP),TP/(TP+FN)

	def _get_score(self,clf,X_train,y_train,X_test,y_test):
		"""获取分类器在数据集上的准确率"""
		print('开始训练')
		clf.fit(X_train,y_train)
		print('开始预测')
		y_pred = clf.predict(X_test)
		# print (y_test)
		# prec = precision_score(y_test, y_pred)
		# reca = recall_score(y_test, y_pred)
		prec,reca = self._get_precision_and_recall(y_test,y_pred)
		return prec,reca

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
			prec,reca = self._get_score(clf,X_train,y_train,X_test,y_test)
			cnt += 1
			print('Split:{}, precision:{:.4f}, recall:{:.4f}'.format(cnt, prec, reca))
			recall += reca
			precision += prec
		p,r = precision/splits, recall/splits
		f = 2*p*r/(p+r)
		return p,r,f



def active_learning(csv_file1,csv_file2):
	"""在数据集上训练分类模型，并在数据集上进行预测，标出预测错误的样本，以便人工查看"""
	X,y = get_dataset(csv_file1,mw2v,myltp,window=2)
	clf = SVC(kernel='rbf')
	clf.fit(X,y)
	y_pred = clf.predict(X)
	right = (y_pred == y)
	with open(csv_file1,'r',newline="",encoding='utf-8') as file_in:
		with open(csv_file2,'w',newline="",encoding='utf8') as file_output:
			reader = csv.reader(file_in)
			writer = csv.writer(file_output)
			for num,row in enumerate(reader):
				if not right[num]:
					writer.writerow(['F']+row)
				else:
					writer.writerow(['T']+row)

if __name__ == '__main__':
	
	
	print ('加载数据集....')

	source_file = 'data/dne/book_person/book_person_ds2.csv'
	
	dataset = DNE_dataset()
	# dataset.prepare(source_file)
	# dataset.build()
	# dataset.save()
	dataset.load(zero_included=True)
	print (dataset.X.shape)
	
	print ('测试模型')

	clf = TSVM('mylib/svm_light_windows64/',kernel=2,gamma=1/665)
	# clf = SVC()
	p,r,f = dataset.get_score(clf,splits=5)
	print (p,r,f)
	# joblib.dump(clf, "data/dne/model/train_model.m")
	# clf = joblib.load("train_model.m")

	

