import re
from nltk import FreqDist, word_tokenize
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
	
def clean_html(html):
	"""
	Copied from NLTK package.
	Remove HTML markup from the given string.

	:param html: the HTML string to be cleaned
	:type html: str
	:rtype: str
	"""
	# see http://stackoverflow.com/questions/26002076/python-nltk-clean-html-not-implemented

	# First we remove inline JavaScript/CSS:
	cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
	# Then we remove html comments. This has to be done before removing regular
	# tags since comments can contain '>' characters.
	cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
	# Next we can remove the remaining tags:
	cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
	# Finally, we deal with whitespace
	cleaned = re.sub(r"&nbsp;", " ", cleaned)
	cleaned = re.sub(r"  ", " ", cleaned)
	cleaned = re.sub(r"  ", " ", cleaned)
	return cleaned.strip()
    
def vectorize_string(doc):
	words = word_tokenize(doc)
	words = [word.lower() for word in words]
	words = [word for word in words if word not in stops]
	fdist = FreqDist(words)
		
	# to address sparsity issues: currently uses dictionaries 
	freqs = [(word, fdist.freq(word)) for word in set(words)]
	return dict(freqs)
	
def vectorize_string_tfidf(doc, idf):
	words = word_tokenize(doc)
	words = [word.lower() for word in words]
	words = [word for word in words if word not in stops]
	fdist = FreqDist(words)
	
	freqs = []
	# to address sparsity issues: currently uses dictionaries 
	for word in set(words):
		try: freqs += [(word, fdist.freq(word) / idf[word])]
		except KeyError: freqs += [(word, fdist.freq(word))]
	return dict(freqs)
	
def get_svec(svecfile, svecdict, word):
	try: 
		linein = svecfile.getline(svecdict[word])
		vect = linein.split()[1:]
		vect = [float(x) for x in vect]
	except KeyError: vect = [0.0] * 300

	return vect

def vectorize_string_svec(doc, svecfile, svecdict):
	words = word_tokenize(doc)
	words = [word.lower() for word in words]
	words = [word for word in words if word not in stops]
	vect = [0.0] * 300
	for word in words:
		try: vect += get_svec(svecfile, svecdict, word)
		except KeyError: continue
		
	return vect
	
