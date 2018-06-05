import os
import numpy as np
import shutil

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
	def __init__(self,root="svm_light_windows64/",
				kernel=2,gamma=1/10,
				kernel_cache=500,
				weight_for_pn=1.,
				C=1.,
				verbosity=1.,
				degree=3,
				remove_inconsistent=0):
		"""
		kernel: 0 linear, 1 poly, 2 rdf, 3 sigmoid.
		kernel_cache: 越大越快 [int 5-正无穷].
		weight_for_pn: 正例和负例的重要度.
		C: 正则项的比重，越大会使得过拟合越严重 [float].
		verbosity: 冗余度 [float 0-3].
		degree: 当核函数为多项式的时候，多项式的次数[int].
		remove_inconsistent: 训练完毕，移除训练集中不一致的样本，重新训练[int 0,1].

		"""
		self.root = root
		self.kernel = kernel
		self.gamma = gamma
		self.kernel_cache = kernel_cache
		self.weight_for_pn = weight_for_pn
		self.C = C
		self.verbosity = verbosity
		self.degree = degree
		self.remove_inconsistent = remove_inconsistent
		self.model_path = None

	def fit(self,X,y):
		exe_path = self.root + 'svm_learn.exe'
		train_path = self.root + 'data/train.dat'
		model_path = self.root + 'data/model'
		kernel_option = '-t %d'%self.kernel
		gamma_option = '-g %f'%self.gamma
		kernel_cache_option = '-m %d'%self.kernel_cache
		weight_for_pn_option = '-j %f'%self.weight_for_pn
		C_option = '-c %f'%self.C
		verbosity_option = '-v %f'%self.verbosity
		degree_option = '-d %d'%self.degree
		remove_inconsistent_option = '-i %d'%self.remove_inconsistent

		# -p
		cmd = '\"%s\" %s %s %s %s %s %s %s %s %s %s'%(exe_path,kernel_option,
													kernel_cache_option,
													weight_for_pn_option,
													C_option,
													verbosity_option,
													gamma_option,
													remove_inconsistent_option,
													degree_option,
													train_path,model_path)
		svmlight_file(train_path).write(X,y)
		a = exec_cmd(cmd)

	def predict(self,X):
		exe_path = self.root + 'svm_classify.exe'
		test_path = self.root + 'data/test.dat'
		pred_path = self.root + 'data/pred'
		if not self.model_path:
			model_path = self.root + 'data/model'
		cmd = '\"%s\" %s %s %s'%(exe_path,test_path,model_path,pred_path)
		svmlight_file(test_path).write(X,y=np.zeros(len(X)))
		exec_cmd(cmd)
		return svmlight_file(pred_path).read()

	def score(self,X,y):
		y_pred = self.predict(X)
		return sum(y_pred == y)/len(y)

	def save(self,path):
		srcfile = self.root + 'data/model'
		shutil.copyfile(srcfile,path)

	def load(self,path):
		self.model_path = path


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
