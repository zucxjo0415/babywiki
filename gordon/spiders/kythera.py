import scrapy
from google import search
from gordon.items import Webpage
from gordon.commons import * 
from math import sqrt
from csv import reader

# for semantic vectors
import linereader

class KytheraSpider(scrapy.Spider):
	name = "kythera"
    # allowed_domains not set, in order to allow all domains
	custom_settings = {'DEPTH_PRIORITY': 1, 'DEPTH_STATS_VERBOSE': True,
			'CLOSESPIDER_PAGECOUNT': 300}

	topic = {}
	start_urls = []
	
	priorityScale = 100		# because scrapy only wants integer priority scores
	threshold = 15			# links with (total) scaled scores above this are crawled
	allWgt = 3				# weight given to parent doc scores rel to anchortext score
	relWgt = 0.3			# weight given to each hi-relevance link (rel to full cosine similarity)
	iridium = {}			# dictionary for collecting linked relevance scores
	
	hiThresh = 0.85			# high-relevance threshold  (not scaled)
	autoThresh = 0.05		# download threshold (not scaled)
	osmium = {}				# dictionary for collecting # hi-relevance links
	crawled = start_urls	# list of crawled urls
	
	idf = {}
	svecFilename = 'C:\cygwin64\home\Zhu Feng\glove.vecs'
	svecdict = {}
	svecfile = None
	mode = ''
	
	def __init__(self, query='tea', topVectRaw='topvect.csv', mode='tf'):
		self.mode = mode
		self.start_urls = list(search(query, stop=20))
        
		if mode =='tfidf':
			with open('..\iphigeni\webidf.csv') as infile:
				vectRead = reader(infile)
				next(vectRead)
				for line in vectRead: self.idf[line[0]] = float(line[1])
		if mode == 'svec':
			with open(self.svecFilename) as infile:
				k = 1
				for line in infile:
					self.svecdict[line.split()[0]] = k
					k += 1
			self.svecfile = linereader.dopen(self.svecFilename)
			
		# Use the veblen spider to generate topVectRaw
		with open(topVectRaw) as infile: 
			vectRead = reader(infile)
			next(vectRead)
			if mode in ['tf', 'tfidf']:
				for line in vectRead:
					try: self.topic[line[1]] += int(line[0])
					except KeyError: self.topic[line[1]] = int(line[0])
				#self.topic = {k: log(v) for k, v in self.topic.items()}
				if mode == 'tfidf':
					for k in self.topic.keys():
						try: self.topic[k] /= self.idf[k]
						except KeyError: continue
			else: 
				self.topic = [0.0] * 300
				for line in vectRead:
					try: 
						svec = get_svec(self.svecfile, self.svecdict, line[1])
						self.topic += [x * int(line[0]) for x in svec]
					except KeyError: continue

	def parse(self, response):
		"""
		The lines below is a spider contract. For more info see:
		http://doc.scrapy.org/en/latest/topics/contracts.html
		
		@url https://www.google.com/search?q=personal+nutrition
		@scrapes pages to depth<=3, using priority-score based BFS
		"""
		
# http://stackoverflow.com/questions/23156780/how-can-i-get-all-the-plain-text-from-a-website-with-scrapy
		try: encoding = response.xpath('//meta/@charset').extract()[0]
		except IndexError: encoding = 'utf-8'
		doc = clean_html(response.body.decode(encoding))

		# turn doc into a tf vector!
		### or use tf-idf features instead
		### or word2vec features (pre-trained on large corpus)
		### e.g. one of the ones from https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-models
		
		try: title = clean_html(response.xpath('//title').extract()[0])
		except IndexError: title = ''
		doc = ' '.join([doc, title, title, title])
		if self.mode == 'tf': parentScore = self.sim(self.topic, vectorize_string(doc))
		elif self.mode == 'tfidf': parentScore = self.sim(self.topic, 
				vectorize_string_tfidf(doc, self.idf))
		else: parentScore = self.sim2(self.topic, 
				vectorize_string_svec(doc, self.svecfile, self.svecdict))
		
		# modify topic vector if we hit an especially relevant page
		if parentScore > self.hiThresh:
			token_counts = vectorize_string(doc)
			for word in token_counts.keys():
				if self.mode in ['tf', 'tfidf']:
					toadd = token_counts[word]
					if self.mode == 'tfidf':
						try: toadd /= self.idf[word]
						except KeyError: toadd /= 1
					try: self.topic[word] += toadd
					except KeyError: self.topic[word] = toadd
				else:
					try: self.topic += get_svecfile(self.svecfile, self.svecdict, word)
					except KeyError: continue
		
		for link in response.xpath("//a"):
			try: url = response.urljoin(link.xpath('@href').extract()[0])
			except IndexError: continue
	
			# add parent (doc, topic) score to forward link
			try: self.iridium[url] += [parentScore]
			except KeyError: self.iridium[url] = [parentScore]		
			# track # links from hi-relevance pages
			if parentScore > self.hiThresh: 
				try: self.osmium[url] += 1
				except KeyError: self.osmium[url] = 1
			# avoid issuing multiple requests for same page
			if url in self.crawled: continue
			
			# get anchor text similarity score, where possible
			try: 
				linktext = link.xpath('text()').extract()[0]
				try: 
					if self.mode == 'tf': anchorScore = self.sim(self.topic, vectorize_string(linktext))
					elif self.mode == 'tfidf': anchorScore = self.sim(self.topic, vectorize_string_tfidf(linktext, self.idf))
					else: anchorScore = self.sim2(self.topic, 
							vectorize_string_svec(linktext, self.svecfile, self.svecdict))
				except ZeroDivisionError: anchorScore = 0.0
			except IndexError: anchorScore = 0.0
			
			# aggregate priority score
			aggParScore = sum(self.iridium[url]) / len(self.iridium[url])
			score = (aggParScore * self.allWgt + anchorScore) / (self.allWgt + 1)
			if url in self.osmium.keys(): score = score + self.osmium[url] * self.relWgt
			score = int(score * self.priorityScale)
			
			if score > self.threshold: 
				self.crawled += [url]
				yield scrapy.Request(url, callback=self.parse, priority=score)
				item = Webpage()
				item['url'] = url
				item['score'] = score
				yield item
		
		if parentScore > self.autoThresh:
			filename = response.url.split("/")[-1] + '.html'
			with open(filename, 'wb') as f:
				# no unicode sandwich here, everything goes through raw.
				f.write(response.body)
			
	def sim(self, v, w):
		commonKeys = set(v.keys()).intersection(w.keys())
		norm_v = sqrt(sum([x ** 2 for x in v.values()]))
		norm_w = sqrt(sum([x ** 2 for x in w.values()]))
		inner_vw = sum([v[x] * w[x] for x in commonKeys])

		# and then do the cosine similarity computation
		score = float(inner_vw) / (norm_v * norm_w)
		# or: cosine similarity via numpy!
		# score = numpy.dot(u, v) / (math.sqrt(numpy.dot(u, u)) * math.sqrt(numpy.dot(v, v)))
		
		return score
		
	def sim2(self, v, w):
		norm_v = sqrt(sum([x ** 2 for x in v]))
		norm_w = sqrt(sum([x ** 2 for x in w]))
		inner_vw = sum(p * q for p, q in zip(v, w))

		# and then do the cosine similarity computation
		score = float(inner_vw) / (norm_v * norm_w)
		# or: cosine similarity via numpy!
		# score = numpy.dot(u, v) / (math.sqrt(numpy.dot(u, u)) * math.sqrt(numpy.dot(v, v)))
		
		return (score + 1.0)/2