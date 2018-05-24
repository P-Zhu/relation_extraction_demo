import os
import numpy as np

# os.system()
def exec_cmd(command):
	return os.popen(command).read()

class svmlight_file:
	"""X y 与 文件的转换"""
	def __init__(self,file_path):
		self.file_path = file_path

	def read(self):
		y = []
		with open(self.file_path,'r',newline="",encoding='utf-8') as file_in:
			for line in file_in:
				if not line:
					continue
				label = float(line)
				if label>0:
					y.append(1)
				else:
					y.append(-1)
		return np.array(y)


	def write(self,X,y):
		with open(self.file_path,'w',newline="",encoding='utf-8') as file_out:
			for feature,label in zip(X,y):
				line = [str(label)]
				for index in range(len(feature)):
					f = feature[index]
					if f!=0:
						try:
							line.append('%d:%s'%(index+1,f))
						except:
							line.append('%d:%f'%(index+1,f))
				line = ' '.join(line)+'\n'
				file_out.writelines([line])
		

class TSVM:
	# http://svmlight.joachims.org/
	def __init__(self,root="svm_light_windows64/",kernel=0,gamma=1/10):
		"""
		0 linear
		1 poly
		2 rdf
		3 sigmoid
		"""
		self.root = root
		self.kernel = kernel
		self.gamma = gamma

	def fit(self,X,y):
		exe_path = self.root + 'svm_learn.exe'
		train_path = self.root + 'data/train.dat'
		model_path = self.root + 'data/model'
		kernel_option = '-t %d'%self.kernel
		gamma_option = '-g %f'%self.gamma

		svmlight_file(train_path).write(X,y)
		cmd = '\"%s\" %s %s %s %s'%(exe_path,kernel_option,gamma_option,train_path,model_path)
		a = exec_cmd(cmd)

	def predict(self,X):
		exe_path = self.root + 'svm_classify.exe'
		test_path = self.root + 'data/test.dat'
		pred_path = self.root + 'data/pred'
		model_path = self.root + 'data/model'
		cmd = '\"%s\" %s %s %s'%(exe_path,test_path,model_path,pred_path)
		svmlight_file(test_path).write(X,y=np.zeros(len(X)))
		exec_cmd(cmd)
		return svmlight_file(pred_path).read()

	def score(self,X,y):
		y_pred = self.predict(X)
		return sum(y_pred == y)/len(y)



if __name__ == '__main__':
	from sklearn.datasets import load_breast_cancer
	from sklearn.svm import SVC

	def load_data():
		bc_data = load_breast_cancer()
		X = bc_data['data']
		y = bc_data['target']
		y = y*2-1
		return X,y

	for clf in (TSVM(),SVC()):
		X,y = load_data()
		clf.fit(X,y)
		accu = clf.score(X,y)
		print (accu)
