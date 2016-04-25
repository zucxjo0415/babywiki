from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import os, re
from lxml import html

# sklearn.optimize import issues fixed
# thank you, https://groups.google.com/a/continuum.io/forum/#!topic/anaconda/bArohmkmc2g

n_features = 1000
n_topics = 5
n_top_words = 20

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
	
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

### Load your scraped pages, re-tokenize, and vectorize result.
# We use a few heuristics to filter out useless terms early on: the posts are 
# stripped of headers, footers and quoted replies, and common English words, 
# words occurring in only one document or in at least 95% of the documents are removed.

data = []
for filename in os.listdir(os.getcwd()):
	if filename.endswith('.html'): 
		tree = html.parse(filename)
		try: encoding = tree.xpath('//meta/@charset')[0]
		except IndexError: encoding = 'utf-8'

		with open(filename) as page:
			rawtext = page.read()
			try: rawtext = rawtext.decode(encoding, errors='backslashreplace')
			except TypeError: continue
			# encoding issues, see http://stackoverflow.com/questions/19527279/python-unicode-to-ascii-conversion
			data += [clean_html(rawtext)]
			if not(len(data) % 10): print("loaded " + str(len(data)) + " documents")
# data is a list, each element of which is a string
# containing the (raw) text of an entire document
	
# Use tf-idf features for NMF.
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, #max_features=n_features,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(data)

# Use tf (raw term count) features for LDA.
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(data)

# Fit the NMF model
nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
#exit()

print("\nTopics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online', learning_offset=50.,
                                random_state=0)
lda.fit(tf)

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)