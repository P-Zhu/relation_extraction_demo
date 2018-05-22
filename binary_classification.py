from mylib import MyWord2Vec,myLTP,Vectorization,mySKF
import numpy as np
import random
import csv
import os
import shutil
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.metrics import recall_score,precision_score


def build_dataset(csv_file1,csv_file2):
	""""""
	with open(csv_file1,'r',newline="",encoding='utf-8') as file_in:
		with open(csv_file2,'w',newline="",encoding='utf8') as file_output:
			reader = csv.reader(file_in)
			writer = csv.writer(file_output)
			for row in reader:
				confidence = float(row[0])
				if abs(confidence) < 0.3:
					continue
				if confidence < 0:
					_y = -1
				else:
					_y = 1
				row[0]=_y
				writer.writerow(row)


def get_dataset(csv_file,w2v,ltp,window=2,random_shuffle=False):
	X = []
	y = []
	with open(csv_file,'r',newline="",encoding='utf-8') as file_in:
		reader = csv.reader(file_in)
		for row in reader:
			_y,_,_,words,loc1,loc2,se1,se2,se3 = row
			_y = float(_y)
			loc1 = eval(loc1)
			loc2 = eval(loc2)
			words = eval(words)
			_X = Vectorization(loc1,loc2,words,se1,se2,se3,w2v,ltp,window=window).vec()
			y.append(_y)
			X.append(_X)
	if random_shuffle:
		Xy = list(zip(X,y))
		random.shuffle(Xy)
		X,y = zip(*Xy)
		X,y = np.array(X),np.array(y)
	return np.array(X),np.array(y)


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


def get_score(clf,X,y,splits=5):
	"""获取SVM在数据集上的准确率"""
	precision = 0
	recall = 0
	mskf = mySKF(X,y,splits)
	cnt = 0
	for X_train,y_train,X_test,y_test in mskf:
		clf.fit(X_train,y_train)
		y_pred = clf.predict(X_test)
		prec = precision_score(y_test, y_pred)
		reca = recall_score(y_test, y_pred)
		cnt += 1
		print('Split:{}, precision:{:.4f}, recall:{:.4f}'.format(cnt, prec, reca))
		recall += reca
		precision += prec
	return precision/splits, recall/splits


if __name__ == '__main__':
	mw2v = MyWord2Vec('data/w2v/',size=128)
	mw2v.load()

	myltp = myLTP(r'../ltp-model','mylib/pattern.txt')
	myltp.load([0,1,0,1,0])
	
	print ('加载数据集....')

	source_file = 'data/dne/book_person/book_person_ds2.csv'
	csv_file = 'data/dne/dataset/book_person.csv'
	csv_file_al = 'data/dne/dataset/book_person_al.csv'

	build_dataset(source_file,csv_file)
	# active_learning(csv_file,csv_file_al)
	X,y = get_dataset(csv_file,mw2v,myltp,window=2,random_shuffle=True)
	print (X.shape,y.shape,sum(y)/len(y))

	svm2 = SVC(kernel='rbf')
	for clf in (svm2,):
		p,r = get_score(clf,X,y,splits=5)
		f = 2*p*r/(p+r)
		print (p,r,f)
		# joblib.dump(clf, "train_model.m")
		# clf = joblib.load("train_model.m")

	myltp.release()

