import numpy as np
import copy


# pos_str = 'a b c d e g h i j k m n nd nh ni nl ns nt nz o p q r u v wp ws x z'
pos_dic = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 
		   'g': 5, 'h': 6, 'i': 7, 'j': 8, 'k': 9,
		   'm': 10, 'n': 11, 'nd': 12, 'nh': 13, 'ni': 14,
		   'nl': 15, 'ns': 16, 'nt': 17, 'nz': 18, 'o': 19, 
		   'p': 20, 'q': 21, 'r': 22,'u': 23, 'v': 24, 
		   'wp': 25,'ws': 26, 'x': 27, 'z': 28, '%': 29}

# parse_str = 'SBV VOB IOB FOB DBL ATT ADV CMP COO POB LAD RAD IS HED WP'
parse_dic = {'SBV': 0, 'VOB': 1, 'IOB': 2, 'FOB': 3,
			 'DBL': 4, 'ATT': 5, 'ADV': 6, 'CMP': 7,
			 'COO': 8, 'POB': 9, 'LAD': 10, 'RAD': 11, 'IS': 12, 'HED': 13, 'WP': 14}

class Vectorization:
	"""
	特征：
		1.实体本身     128 * 2
		2.实体的长度   1 * 2
		-------------
		3.实体的上下文 128 * 3
		4.上下文的词性 30 * 3
		-------------
		5.实体间的距离   1
		6.实体的相对位置 1
		7.句法分析       15 * 2
		-------------
		8.点互信息       3
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

		self.pos = self._get_pos()
		self.arcs = self._get_arcs()

	def words_embedding(self,words):
		return self.w2v.words_embedding(words)

	def pos_embedding(self,postags):
		pos_vec = [0 for i in range(30)]
		for pos in postags:
			index = pos_dic[pos]
			pos_vec[index] += 1
		return pos_vec

	def arc_embedding(self,arcs):
		arcs_vec = [0 for i in range(15)]
		arcs = list(self.arcs)
		for arc in arcs:
			index = parse_dic[arc.relation]
			arcs_vec[index] += 1
		return arcs_vec

	def entity_feature(self,loc):
		start,end = loc
		ne_words = self.words[start:end]
		ne_vec  = self.words_embedding(ne_words)
		length = len(''.join(ne_words))
		return ne_vec,length

	def find_context(self):
		context_pre  = []
		context_mid  = []
		context_post = []
		start1,end1 = self.loc1
		start2,end2 = self.loc2 
		if start1>start2:
			start1,start2 = start2,start1
			end1,end2 = end2,end1
		for i in range(len(self.words)):
			if start1-self.window<=i<start1:
				context_pre.append(i)
			elif end1<=i<start2:
				context_mid.append(i)
			elif end2<=i<end2+self.window:
				context_post.append(i)
		return context_pre,context_mid,context_post
				
	def context_feature(self,context):
		context_words = [self.words[index] for index in context]
		context_pos   = [self.pos[index] for index in context]
		
		context_vec  = self.words_embedding(context_words)
		pos_vec = self.pos_embedding(context_pos)
		return context_vec,pos_vec

	def sentence_feature(self):
		start1,end1 = self.loc1
		start2,end2 = self.loc2
		d = min(abs(start1 - end2),abs(start2 - end1))
		rp = int(start1<end2) # 相对位置

		arc1 = self.arcs[start1:end1]
		arc2 = self.arcs[start2:end2]
		arc_vec1 = self.arc_embedding(arc1)
		arc_vec2 = self.arc_embedding(arc2)
		
		return d,rp,arc_vec1,arc_vec2

	def _get_pos(self):
		start,end = self.loc2
		words = copy.deepcopy(self.words)
		words[start:end] = '书'
		return self.ltp.pos_tag(words)

	def _get_arcs(self):
		start,end = self.loc2
		words = copy.deepcopy(self.words)
		words[start:end] = '书'
		return self.ltp.parse(words, self.pos)

	def vec(self):
		ne_vec1,length1 = self.entity_feature(self.loc1)
		ne_vec2,length2 = self.entity_feature(self.loc2)

		c1,c2,c3 = self.find_context()
		c_vec1,pos_vec1 = self.context_feature(c1)
		c_vec2,pos_vec2 = self.context_feature(c2)
		c_vec3,pos_vec3 = self.context_feature(c3)

		self.d,self.rp,arc_vec1,arc_vec2 = self.sentence_feature()
		return np.concatenate((ne_vec1,ne_vec2,
					c_vec1,c_vec2,c_vec3,
					pos_vec1,pos_vec2,pos_vec3,
					arc_vec1,arc_vec2,
					(length1,length2,self.d,self.rp,
					 self.se1,self.se2,self.se3)))

	def pvec(self):
		return self.vec(),self.pos_tag,self.arcs,self.d,self.rp



if __name__ == "__main__":
	Vectorization(**arg).vec()
