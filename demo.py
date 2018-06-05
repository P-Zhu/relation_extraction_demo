from mylib import myLTP,TSVM,MyWord2Vec,myLTP,Vectorization,baidu_result_num
import math



class Find_Author_Title:
	def __init__(self,sentences,myltp,mw2v,model):
		self.sentences = sentences
		self.myltp = myltp
		self.mw2v = mw2v
		self.model = model

	def main(self):
		ne_pairs = self.get_ne_pair()
		for ne_pair in ne_pairs:
			feature,book,person,words,loc1,loc2 = self.get_feature(ne_pair)
			isa = self.is_author(feature)
			print (isa,person,book,words)


	def get_ne_pair(self):
		ne_pairs = []
		results = self.myltp._get_dne_for_sentence(sentences)
		for r in results:
			ne1,ne2,words=r
			if ne1>ne2:
				ne1,ne2 = ne2,ne1
			if (ne1[0],ne2[0]) == ('BOOK','Nh'):
				ne_pairs.append((ne1,ne2,words))
		return ne_pairs

	def get_feature(self,ne_pair):
		ne1,ne2,words = ne_pair
		_,start1,end1 = ne1
		_,start2,end2= ne2

		book   = ''.join(words[start1:end1])
		person = ''.join(words[start2:end2])

		person_result_num = baidu_result_num(person)
		book_result_num = baidu_result_num(book)
		joint_num1 = baidu_result_num("%s %s"%(person,book))
		joint_num2 = baidu_result_num("%s %s"%(book,person))
		se1 = math.log(person_result_num+1,10)
		se2 = math.log(book_result_num+1,10)
		se3 = (math.log(joint_num1+1,10)*math.log(joint_num1+1,10))**0.5
		feature = Vectorization((start1,end1),(start2,end2),words,se1,se2,se3,self.mw2v,self.myltp,window=2).vec()	
		return feature,book,person,words,(start1,end1),(start2,end2)

	def is_author(self,feature):
		return self.model.predict([feature])
		

if __name__=="__main__":
	myltp = myLTP(r'../ltp-model','mylib/pattern.txt')
	myltp.load([0,1,1,1,0])
	mw2v = MyWord2Vec('data/w2v/',size=128)
	mw2v.load()
	model = TSVM('mylib/svm_light_windows64/',kernel=2,
				gamma=1/665,kernel_cache=1000,
				weight_for_pn=1.,C=2.,verbosity=0.5)
	# model.load()

	sentences = '关于这一点，只要检索袁英光与刘寅生编著的《王国维年谱长编》，就不难发现父亲王乃誉"饬静儿"之语比比皆是。'
	FAT = Find_Author_Title(sentences,myltp,mw2v,model)
	FAT.main()

	myltp.release()
