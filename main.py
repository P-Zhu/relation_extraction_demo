from mylib import MyWord2Vec


if __name__ == '__main__':
	mw2v = MyWord2Vec('data/w2v/',size=128)
	# mw2v.seg()
	mw2v.train()
	mw2v.save()