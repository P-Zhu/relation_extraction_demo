from mylib import MyWord2Vec,myLTP
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics import recall_score,precision_score
import csv
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression



class my_SKF:
	def __init__(self, X, y, splits):
		"""用于交叉验证"""
		self.X = X
		self.y = y
		self.skf = StratifiedKFold(n_splits=splits)
		self.skf = self.skf.split(X, y)

	def __next__(self):
		train_index, test_index = self.skf.__next__()
		X_train, X_test = self.X[train_index], self.X[test_index]
		y_train, y_test = self.y[train_index], self.y[test_index]
		return X_train, y_train, X_test, y_test

	def __iter__(self):
		return self


# pos_str = 'a b c d e g h i j k m n nd nh ni nl ns nt nz o p q r u v wp ws x z'
pos_dic = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 
		   'g': 5, 'h': 6, 'i': 7, 'j': 8, 'k': 9,
		   'm': 10, 'n': 11, 'nd': 12, 'nh': 13, 'ni': 14,
		   'nl': 15, 'ns': 16, 'nt': 17, 'nz': 18, 'o': 19, 
		   'p': 20, 'q': 21, 'r': 22,'u': 23, 'v': 24, 
		   'wp': 25,'ws': 26, 'x': 27, 'z': 28}

# parse_str = 'SBV VOB IOB FOB DBL ATT ADV CMP COO POB LAD RAD IS HED WP'
parse_dic = {'SBV': 0, 'VOB': 1, 'IOB': 2, 'FOB': 3,
			 'DBL': 4, 'ATT': 5, 'ADV': 6, 'CMP': 7,
			 'COO': 8, 'POB': 9, 'LAD': 10, 'RAD': 11, 'IS': 12, 'HED': 13, 'WP': 14}

class Vertorization:
	"""
	特征：
		1.实体本身 
		2.实体的词性
		3.实体的上下文
		4.上下文的词性
		5.实体间的距离
		6.实体的相对位置
		7.搜索相似度
		8.句法分析
	"""
	def __init__(self,loc1,loc2,words,se1,se2,se3,w2v,ltp,window=2):
		self.loc1 = loc1
		self.loc2 = loc2
		self.words = words
		self.window = window
		self.w2v = w2v
		self.ltp = ltp
		self.se1 = se1
		self.se2 = se2
		self.se3 = se3

		start1,end1 = loc1
		start2,end2 = loc2

		self.ne1 = ''.join(words[start1:end1])
		self.ne2 = ''.join(words[start2:end2])
		self.pos = self._get_pos()
		self.arcs = self._get_arcs()
		
		self.ne_vec1 = self._build_ne_vec(self.ne1)
		self.ne_vec2 = self._build_ne_vec(self.ne2)
		self.ne_pos_vec1 = self._build_ne_pos_vec(loc1)
		self.ne_pos_vec2 = self._build_ne_pos_vec(loc2)
		self.ne_arcs_vec1 = self._build_ne_arcs_vec(loc1)
		self.ne_arcs_vec2 = self._build_ne_arcs_vec(loc2)

		context_indexs1 = self._get_context_indexs(loc1,loc2)
		context_indexs2 = self._get_context_indexs(loc2,loc1)
		self.context_vec1 = self._build_context_vec(context_indexs1)
		self.context_vec2 = self._build_context_vec(context_indexs2)
		self.context_pos_vec1 = self._build_context_pos_vec(context_indexs1)
		self.context_pos_vec2 = self._build_context_pos_vec(context_indexs2)

		self.distance = self._get_distance()
		self.relative_position = self._get_relative_position()

	def _get_context_indexs(self,loc,other_loc):
		"""获取一个实体的上下文"""
		context_indexs = []
		start,end = loc
		_star,_en = other_loc
		for i in range(len(self.words)):
			if abs(i-start)<=self.window or abs(i-end-1)<=self.window:
				if (not start<=i<end) and (not _star<=i<_en):
					context_indexs.append(i)
		return context_indexs

	def _get_pos(self):
		return self.ltp.pos_tag(self.words)

	def _get_arcs(self):
		return self.ltp.parse(self.words, self.pos)

	def _build_ne_vec(self,ne):
		return self.w2v.word_embedding(ne)

	def _build_ne_pos_vec(self,loc):
		pos_vec = [0 for i in range(29)]
		start,end = loc
		for i in range(start,end):
			index = pos_dic[self.pos[i]]
			pos_vec[index] += 1
		return pos_vec

	def _build_ne_arcs_vec(self,loc):
		arcs_vec = [0 for i in range(15)]
		arcs = list(self.arcs)
		start,end = loc
		for i in range(start,end):
			index = parse_dic[arcs[i].relation]
			arcs_vec[index] += 1
		return arcs_vec

	def _build_context_vec(self,context_indexs):
		context = [self.words[index] for index in context_indexs]
		return self.w2v.words_embedding(context)

	def _build_context_pos_vec(self,context_indexs):
		pos_vec = [0 for i in range(29)]
		for i in context_indexs:
			index = pos_dic[self.pos[i]]
			pos_vec[index] += 1
		return pos_vec

	def _get_distance(self):
		# [s1,e1] [s2,e2]
		start1,end1 = self.loc1
		start2,end2 = self.loc2
		return min(abs(start1 - end2),abs(start2 - end1))

	def _get_relative_position(self):
		# [s1,e1] [s2,e2]
		start1,end1 = self.loc1
		start2,end2 = self.loc2
		if start1<end2:
			return 1
		else:
			return -1

	def vec(self):
		return np.concatenate((self.ne_vec1,
				self.ne_vec2,
				self.ne_pos_vec1,
				self.ne_pos_vec2,
				self.ne_arcs_vec1,
				self.ne_arcs_vec2,
				self.context_vec1,
				self.context_vec2,
				self.context_pos_vec1,
				self.context_pos_vec2,
				[self.distance,self.relative_position,
				 self.se1,self.se2,self.se3]))
		

def get_dataset(csv_file,w2v,ltp,window=2):
	X = []
	y = []
	with open(csv_file,'r',newline="",encoding='utf-8') as file_in:
		reader = csv.reader(file_in)
		for row in reader:
			confidence,_,_,words,loc1,loc2,se1,se2,se3 = row
			confidence = float(confidence)
			if abs(confidence) < 0.3:
				continue
			if confidence<0:
				_y = 0
			else:
				_y = 1
			loc1 = eval(loc1)
			loc2 = eval(loc2)
			words = eval(words)
			_X = Vertorization(loc1,loc2,words,se1,se2,se3,w2v,ltp,window=window).vec()
			y.append(_y)
			X.append(_X)
	return np.array(X),np.array(y)
	

def get_score(clf,X,y,splits=5):
	"""获取SVM在数据集上的准确率"""
	precision = 0
	recall = 0
	mskf = my_SKF(X,y,splits)
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
	
	print ('加载数据集')
	csv_file = 'data/dne/book_person/book_person_ds2.csv'
	X,y = get_dataset(csv_file,mw2v,myltp,window=1)

	print (X.shape,y.shape)
	Xy = list(zip(X,y))
	random.shuffle(Xy)
	X,y = zip(*Xy)
	X,y = np.array(X),np.array(y)

	svm2 = SVC(kernel='rbf')
	for clf in (svm2,):
		p,r = get_score(clf,X,y,splits=5)
		f = 2*p*r/(p+r)
		print (p,r,f)

	myltp.release()

