import numpy as np


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
		pos_vec = [0 for i in range(30)]
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


if __name__ == "__main__":
	Vectorization(**arg).vec()
