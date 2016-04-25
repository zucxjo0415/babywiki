import scrapy
from google import search
from gordon.items import WordCount
from gordon.commons import clean_html
from nltk import FreqDist, word_tokenize
from nltk.corpus import stopwords
#from bs4 import BeautifulSoup
from math import sqrt

class VeblenSpider(scrapy.Spider):
	name = "veblen"
    # allowed_domains not set, in order to allow all domains
	# allowed_domains = ["wikipedia.org"]    
	custom_settings = {'DEPTH_PRIORITY': 10, 'DEPTH_STATS_VERBOSE': True,
		'CLOSESPIDER_PAGECOUNT': 60, 'DEPTH_LIMIT': 1}
	start_urls = []
	
	stops = set(stopwords.words('english'))
	
	### you should learn how to use the scrapy shell, it'll be mad useful for debugging.
	def __init__(self, query='tea'):
		self.start_urls = list(search(query, stop=20))
        
	def parse(self, response):
		"""
		The lines below is a spider contract. For more info see:
		http://doc.scrapy.org/en/latest/topics/contracts.html
		
		@url https://www.google.com/search?q=personal+nutrition
		@scrapes pages to depth<=3, using priority-score based BFS
		"""
		
		doc = clean_html(response.body_as_unicode())
		words = word_tokenize(doc)
		words = [word.lower() for word in words]
		words = [word for word in words if word not in self.stops]
		fdist = FreqDist(words)
		
		for word in set(words):
			if (fdist.freq(word) * fdist.N()) > 1:
				item = WordCount()
				item['word'] = word
				item['count'] = int(fdist.freq(word) * fdist.N())
				yield item 
		#for href in response.css("a::attr('href')"):
		#	url = response.urljoin(href.extract())
		#	yield scrapy.Request(url, callback=self.parse)