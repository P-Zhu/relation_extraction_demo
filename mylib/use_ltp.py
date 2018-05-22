import os
import re
import jieba
from collections import defaultdict
from itertools import combinations
from pyltp import SentenceSplitter,\
				  Segmentor,Postagger,\
				  NamedEntityRecognizer,\
				  Parser,\
				  SementicRoleLabeller


# BOOK类别
# 时间、时间段
# \d{1,2}世纪(\d0年代|中期|中叶|以后|初|末)?
# \d{1,4}年(\d{1,2}月)?
# (\d{1,4}年)?(\d{1,2}月)\d{1,2}日


class myLTP:
	def __init__(self,LTP_DATA_DIR,pattern_dir='pattern.txt'):
		self.LTP_DATA_DIR = LTP_DATA_DIR
		self.ne_pattern = self._read_ne_pattern(pattern_dir)		

	def _read_ne_pattern(self,filename):
		ne_pattern = []
		with open(filename,encoding='utf8') as filein:
			for line in filein:
				if line[0] != '#':
					np=line.split()[:2]
					ne_pattern.append(np)
		return ne_pattern

	def find_ne_by_pattern(self,text):
		ne_dic = defaultdict(list)
		for ne_type,pattern in self.ne_pattern:
			nes = re.findall(pattern,text)
			text = re.sub(pattern,ne_type,text)
			ne_dic[ne_type].extend(nes)
		return text,ne_dic

	def load(self,index=[1,1,1,1,1]):
		"""分词 词性标注 命名实体识别 句法分析 语义角色分析"""
		if index[0]:
			cws_model_path = os.path.join(self.LTP_DATA_DIR, 'cws.model')
			self.segmentor = Segmentor()
			self.segmentor.load(cws_model_path)

		if index[1]:
			pos_model_path = os.path.join(self.LTP_DATA_DIR, 'pos.model')
			self.postagger = Postagger()
			self.postagger.load(pos_model_path)

		if index[2]:
			ner_model_path = os.path.join(self.LTP_DATA_DIR, 'ner.model') 
			self.recognizer = NamedEntityRecognizer()
			self.recognizer.load(ner_model_path)

		if index[3]:
			par_model_path = os.path.join(self.LTP_DATA_DIR, 'parser.model')
			self.parser = Parser() 
			self.parser.load(par_model_path)  

		if index[4]:
			srl_model_path = os.path.join(self.LTP_DATA_DIR, 'pisrl_win.model')
			self.labeller = SementicRoleLabeller() 
			self.labeller.load(srl_model_path) 
		
	def release(self):
		try: 
			self.segmentor.release()
		except: 
			pass
		try: 
			self.postagger.release()
		except: 
			pass
		try: 
			self.recognizer.release()
		except: 
			pass
		try:
			self.parser.release()
		except: 
			pass
		try: 
			self.labeller.release()
		except: 
			pass

	def split_sentence(self,text):
		"""分句"""
		return SentenceSplitter.split(text)

	def word_segment(self,sentence):
		"""使用结巴分词"""
		# words = self.segmentor.segment(sentence)
		words = jieba.cut(sentence)
		return words

	def pos_tag(self,words):
		"""词性标注"""
		postags = self.postagger.postag(words)
		return postags

	def named_entity_recognize(self,words,postags):
		"""命名实体识别"""
		netags = self.recognizer.recognize(words, postags)
		return netags

	def parse(self,words,postags):
		"""句法分析"""
		arcs = self.parser.parse(words, postags) # (arc.head, arc.relation)
		return arcs

	def sementic_role_label(self,words,postags,arcs):
		"""语义角色分析"""
		roles = self.labeller.label(words, postags, arcs)
		return roles

	def _get_ne_for_sentence(self,sentence):
		"""获取实体，包括通过正则表达式定义的一些实体"""
		
		sentence,ne_dic = self.find_ne_by_pattern(sentence)
		words = list(self.word_segment(sentence))
		postags = self.postagger.postag(words)
		ners = self.named_entity_recognize(words,postags)
		res = {}
		res['words'] = words
		res['ners'] = []
		for index,ner in enumerate(ners):
			if ner != 'O':
				if ner[0] in ('S','B'):
					res['ners'].append([ner[2:],index,index+1])
				else:
					res['ners'][-1][-1] += 1	
		for ner_type,v in ne_dic.items():
			v = iter(v)
			if v:
				for index,word in enumerate(words):
					if word == ner_type:
						words[index] = v.__next__()
						res['ners'].append((ner_type,index,index+1))
		return res

	def _get_dne_for_sentence(self,sentence):
		res = []
		s = self._get_ne_for_sentence(sentence)
		ners = s['ners']
		words = s['words']
		for entity1,entity2 in combinations(ners,2):
			res.append((entity1,entity2,words))
		return res

	def get_dne(self,text):
		"""获取实体对，人名(Nh)地名(Ns)机构名(Ni)"""
		res = []
		sentences = self.split_sentence(text)
		for sentence in sentences:
			r = self._get_dne_for_sentence(sentence)
			res.extend(r)
		return res


if __name__ == "__main__":
	
	myltp = myLTP(r'../../../../../../../../../_mass/ltp-model')
	myltp.load([1,1,1,1,0])
	
	words = ['①', '张德坚', '初撰', '《贼情集要》', '，', '后', '至', '曾', '国藩', '所', '设', '采编', '所', '，', '一八五五', '年', '成', '《贼情汇纂》', '。']
	postags = myltp.pos_tag(words)
	res = myltp.parse(words,postags)
	res = list(res)
	for i in range(len(res)):
		print (res[i].head,res[i].relation)
	print (i,len(words))
	myltp.release()

