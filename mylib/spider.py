"""
find_author(book,person)
远程监督判断<文章,作者>是否正确。

监督源：
1. 百度百科 0.8
2. 文津搜索 0.4
3. wikisource 1
4. 百科搜索结果TITLE 1

"""
from bs4 import BeautifulSoup
from urllib import request,parse
from selenium import webdriver
from  collections import defaultdict
import time
import urllib
import requests
import re
import random
import math
try:
	from mylib.langconv import *
except:
	from langconv import *

my_headers=["Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0",
			"Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)",
			'Mozilla/5.0 (Windows NT 6.2; WOW64; rv:21.0) Gecko/20100101 Firefox/21.0',
			'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.94 Safari/537.36',
			'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.106 Safari/537.36',
			'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; LBBROWSER) ',  
			'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; QQBrowser/7.0.3698.400)'
			]

def cp2ep(word):
	# 部分 中文标点转英文标点
	lst = [('，',','),(')','）'),('(','（'),('：',':')]
	for c,e in lst:
		word = word.replace(c,e)
	return word

def remove_punc(word):
	word = re.sub('[《》“”‘’<>]','',word)
	return cp2ep(word)

def context_equal(w1,w2):
	return remove_punc(w1) == remove_punc(w2)

def t2s(text):
	return Converter('zh-hans').convert(text)

def download(url,args={}):
	# random_header = random.choice(my_headers)
	# if args:
	# 	args = parse.urlencode(args).encode('utf-8')
	# 	req = request.Request(url,data=args)
	# else:
	# 	req = request.Request(url)
	# req.add_header('User-Agent',random_header)
	# with request.urlopen(req) as f:
	# 	data = f.read().decode('utf-8',errors='ignore')
	random_header = random.choice(my_headers)
	headers = {'User-Agent':random_header}
	data = requests.get(url, headers=headers, params=args)
	data.encoding = 'utf-8'
	return data.text

def download_just_baike(url):
	real_url = requests.head(url).headers['location']
	if 'baike.baidu.com' not in real_url:
		return None
	random_header = random.choice(my_headers)
	headers = {'User-Agent':random_header}
	data = requests.get(real_url, headers=headers)
	data.encoding = 'utf-8'
	return data.text



def get_info_from_html_for_baike(html,url):
	soup = BeautifulSoup(html,'html.parser')
	title = soup.find('dl',class_='lemmaWgt-lemmaTitle')
	if not title:
		# print (url)
		return
	title = title.find('h1').get_text()
	keys = soup.find_all('dt',class_='basicInfo-item name')
	values = soup.find_all('dd',class_='basicInfo-item value')
	infor_dic = defaultdict(int)
	for i in range(len(keys)):
		key = keys[i].get_text().replace('\xa0','')
		value = values[i].get_text().replace('\n','')
		infor_dic[key] = value
	tag = soup.find('div',id='open-tag')
	if tag:
		tags = tag.get_text().replace('\n','')[5:].split('，')
	else:
		tags = []
	
	return infor_dic,title,tags


def find_author_from_wikisource(book):
	book = remove_punc(book)
	url = 'https://zh.wikisource.org/wiki/'+request.quote(book)
	try:
		html = download(url)
	except urllib.error.HTTPError as e: # 404
		assert e.code == 404
		return 0
	soup = BeautifulSoup(html,'html.parser')
	table = soup.find('table',style='width:100%; margin-top:0px;border:1px solid #93A6C0; background-color: #F9F9F9; text-align:center;')
	
	if not table:
		return 0
	try:
		text = table.find_all('td')[2].get_text()
	except IndexError:
		print ('\t\t\tIndexError:',book)
		return 0

	author = re.findall(r'作者\S{1,30}[　 \n]',text)
	if author:
		return author[0].replace('\n','')
	else:# 未统计作者
		return 0
	

baidu_search_uri = lambda w:'http://www.baidu.com/s?ie=UTF-8&wd='+request.quote(w)

def to_baike_url(word):
	html = download(baidu_search_uri('百度百科 '+word))
	soup = BeautifulSoup(html,'html.parser')
	links = [node.find('a') for node in soup.find_all('h3',class_="t")]# t c-gap-bottom-small
	return [link['href'] for link in links[:5]]


def find_author_by_title(book,person):
	book = remove_punc(book)
	url  = baidu_search_uri(person+' '+book)
	# print(url)
	html = download(url)
	soup = BeautifulSoup(html,'html.parser')# t c-gap-bottom-small
	h3s  = soup.find_all('h3',class_="t")
	count = 0
	# print (len(h3s))
	for h3 in h3s:
		h3text = cp2ep(h3.find('a').get_text())
		if re.findall(r''+person+'[的:： \-]?[《]?'+book+'[》_ ]',h3text) or \
			   re.findall(r'[《]?'+book+'[》_ ]'+person,h3text):
				count += 1
	sum_=len(h3s)
	if len(h3s) in (0,1):
		# print ('h3s',book,person)
		return 1,0
	return 1,math.log(count+1,sum_)
	



def find_author_by_wenjin(book):
	book = book.replace('《','').replace('》',"")
	# 文津搜索 - 中国国家图书馆
	# http://find.nlc.cn/search/doSearch&query=%E6%95%AC%E5%91%8A%E9%9D%92%E5%B9%B4&docType=%E5%9B%BE%E4%B9%A6&targetFieldLod=%E9%A2%98%E5%90%8D
	# 问题：容易将责任人 作为 作者
	url_get_base = 'http://find.nlc.cn/search/doSearch'
	args = {'actualQuery':book,
			'docType':'图书',
			'targetFieldLod':'题名'}
	html = download(url_get_base,args)
	soup = BeautifulSoup(html,'html.parser')
	items= soup.find_all('div',class_='info')
	if not items:
		return 0
	item = items[0]	
	str_ = item.get_text()
		
	# str_ = str_.replace('\n\n','\n')
	title  = item.find('h4').get_text().replace('\t','').replace('\n','')
	author = re.findall(r'著者：\n\t\t\S{1,16}\n',str_)
	if context_equal(title,book):
		if author:
			return author[0].replace('\n\t\t','').replace('\n','')
	return 0

journal_tags = ["期刊",'报纸','电视剧']
journal_info = ['创刊地点','创刊时间','制定者']
author_mark = ['作者','编撰者','主要编撰者','发表人物']
def find_author_by_baike(book):
	urls=to_baike_url(book)
	for url in urls:
		try:
			html=download_just_baike(url)
			if not html:
				continue
		except urllib.error.HTTPError as e:
			print (book,url,e.code)
			continue
			
		result=get_info_from_html_for_baike(html,url)
		if not result: continue
		dic,title,tags = result
		if context_equal(title,book):
			for jinfo in journal_info:
				if jinfo in dic:
					return -1
			for jtag in journal_tags:
				if jtag in tags:
					return -1
			author  = 0
			for mark in author_mark:
				author = author or dic[mark]
			if author:
				return author
	return 0

def find_author(book,person):
	# 作者 -1 不是图书 ; 0  不确定
	weight = [1,0.8,0.4]
	funs = [find_author_from_wikisource,
			find_author_by_baike,
			find_author_by_wenjin]
	authors = []
	for i in range(len(funs)):
		author = funs[i](book)
		authors.append([author,weight[i]])
	try:
		authors.append(find_author_by_title(book,person))
	except: # 可能会出现正则的错误
		pass
	for i in range(len(authors)):
		if type(authors[i][0]) != int:
			if t2s(person) in t2s(authors[i][0]):
				authors[i][0] = 1
			else:
				authors[i][0] = -1
	
	author = sum([a*w for a,w in authors])
	return author
	

def baidu_result_num(key):
	html = download(baidu_search_uri(key))
	soup = BeautifulSoup(html,'html.parser')
	node = soup.find('div',class_="nums")
	num  = node.get_text()[15:-1].replace(',','')
	return int(num)


if __name__=="__main__":
	import math
	for p,b in (('李大钊','《社会革命底商榷》'),
				('毛泽东','《唯心历史观的破产》')):
		r1 = baidu_result_num(p)
		r2 = baidu_result_num(b)
		r3 = baidu_result_num("%s %s 作者"%(p,b))
		print (math.log(r1+1,10))
		print (math.log(r2+1,10))
		print (math.log(r3+1,10))


