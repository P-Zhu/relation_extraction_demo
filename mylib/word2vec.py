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

def wordSeg_for_xml(in_filename,out_filename,tags=('content','introduction')):
	with open(out_filename,'w+',encoding='utf8') as fileOutput:
		DOMTree=XML.parse(in_filename)
		for tag in tags:
			Data=DOMTree.documentElement.getElementsByTagName(tag)
			for data in Data:
				if data.hasChildNodes():
					print (cut(data.childNodes[0].data)[:50])
					fileOutput.writelines(cut(data.childNodes[0].data))

class MyWord2Vec:
	# 如果使用继承的方法，应该可以写的更好
	def __init__(self,root_dir):
		# 目录格式
		# root_dir/
		# 			txt/
		# 			seg/
		# 			vec/	
		self.model=None
		self.root_dir=root_dir

	def seg(self,xml):
		try:
			shutil.rmtree(self.root_dir+'seg')
		except FileNotFoundError:
			pass
		os.makedirs(self.root_dir+'seg')
		for filename in os.listdir(self.root_dir+'txt'):
			if xml:
				wordSeg_for_xml(self.root_dir+'txt/'+filename,
								 self.root_dir+'seg/'+filename)
			else:
				wordSeg_for_text(self.root_dir+'txt/'+filename,
								 self.root_dir+'seg/'+filename)
	
	def train(self,size=50,xml=False,re_seg=False):
		if not os.path.exists(self.root_dir+'seg') or not os.listdir(self.root_dir+'seg') or re_seg:
			print ("正在进行分词")
			self.seg(xml)
		print ("正在训练词向量")
		sentences = word2vec.PathLineSentences(self.root_dir+'seg/')
		self.model= word2vec.Word2Vec(sentences,size=50) 

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
			print ('查无此词：[%s]'%key)
			return 0

	def pharse_embedding(self,pharse):
		ws = cut(pharse).split()
		vec= []
		for w in ws:
			v = self[w]
			if type(v) != int:
				vec.append(v)
		if len(vec)>0:
			return sum(vec)/len(vec)
		else:
			return np.zeros(50)

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
	w2v.load()
	print (w2v.pharse_embedding('《党的机会主义史》,，:：'))