try:
	from mylib.use_ltp  import myLTP
except:
	pass
from mylib.spider import find_author,baidu_result_num
from mylib.langconv import *
from mylib.word2vec import MyWord2Vec
from mylib.vectorization import Vectorization
from mylib.base_tech import mySKF
from mylib.tsvm import TSVM
from mylib.metric_learning import siamese_network