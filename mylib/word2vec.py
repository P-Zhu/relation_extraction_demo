from gensim.models import word2vec
import numpy as np
import xml.dom.minidom as XML
import os
import shutil
import jieba



def cut(text):
	return ' '.join(jieba.cut(text))

def wordSeg_for_txt(in_filename,out_filename):
	with open(in_filename,encoding='utf8') as fileInput:
		with open(out_filename,'w+',encoding='utf8') as fileOutput:
			for line in fileInput:
				fileOutput.writelines(cut(line))

class MyWord2Vec:
	# 如果使用继承的方法，应该可以写的更好
	def __init__(self,root_dir,size=64):
		# 目录格式
		# root_dir/
		# 			txt/
		# 			seg/
		# 			vec/	
		self.size = size
		self.model=None
		self.root_dir=root_dir

	def seg(self):
		print ("正在进行分词")
		try:
			shutil.rmtree(self.root_dir+'seg')
		except FileNotFoundError:
			pass
		os.makedirs(self.root_dir+'seg')
		for filename in os.listdir(self.root_dir+'txt'):
			wordSeg_for_txt(self.root_dir+'txt/'+filename,
								 self.root_dir+'seg/'+filename)
	
	def train(self):
		print ("正在训练词向量")
		sentences = word2vec.PathLineSentences(self.root_dir+'seg/')
		self.model= word2vec.Word2Vec(sentences,size=self.size) 

	def save(self):
		if not os.path.exists(self.root_dir+'vec'):
			os.makedirs(self.root_dir+'vec')
		self.model.save(self.root_dir+'vec/vec.model')

	def load(self):
		self.model=word2vec.Word2Vec.load(self.root_dir+'vec/vec.model')

	def __getitem__(self,key):
		try:
			return self.model[key]
		except KeyError:
			return np.zeros(self.size)

	def word_embedding(self,word):
		# 如果单词未出现，则尝试把它当成短语
		vec = self[word]
		if not (vec == np.zeros(self.size)).all():
			return vec
		else:
			return self._pharse_embedding(word)

	def words_embedding(self,words):
		if not words:
			return np.zeros(self.size)
		vec= []
		for word in words:
			v = self.word_embedding(word)
			vec.append(v)
		return sum(vec)/len(vec)

	def _pharse_embedding(self,pharse):
		# 为防止循环调用，设置此方法
		words = cut(pharse).split()
		vec= []
		for word in words:
			v = self[word]
			vec.append(v)
		return sum(vec)/len(vec)
		
	def most_similar(self,key):
		return self.model.most_similar(key)

	def similarity(self,key1,key2):
		return self.model.similarity(key1,key2)

def cosine(x,y):
	# 余弦相似性
	return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

if __name__ == "__main__":
	w2v=MyWord2Vec('../data/w2v/')
	# w2v.train()
	# w2v.save()
	# w2v.load()
	print (w2v.pharse_embedding('《党的机会主义史》,，:：'))