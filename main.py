from mylib import MyWord2Vec,myLTP
from sklearn.svm import SVC

from get_ne import find_dne_for_dir,class_dne_by_type
from distance_supervision import prepare_file,\
						distance_supervision,\
						adding_baidu_result_num
from binary_classification import get_dataset



def get_named_entity_pairs(ltp):
	find_dne_for_dir('data/txt/','data/dne/',myltp)
	class_dne_by_type('data/dne/','data/dne/by_type/')

def get_label():
	prepare_file()
	distance_supervision()
	adding_baidu_result_num()

def classification():
	print ('加载数据集')
	csv_file = 'data/dne/book_person/book_person_ds2.csv'
	X,y = get_dataset(csv_file,mw2v,myltp)
	svm1 = SVC(kernel='poly')
	for clf in (svm1,):
		p,r = get_score(clf,X,y)
		f = 2*p*r/(p+r)
		print (p,r,f)


if __name__ == '__main__':
	# 加载 ltp
	myltp = myLTP(r'../ltp-model','mylib/pattern.txt')
	myltp.load([1,1,1,0,0])

	# 记载词向量
	mw2v = MyWord2Vec('data/w2v/',size=128)
	mw2v.load()

	## 从文本中提取实体对
	# get_named_entity_pairs(myltp)

	## 使用远程监督的方法，进行类别标记 
	# get_label()
	
	# 使用SVM进行预测
	classification()
	
	myltp.release()