#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python

# onlinewikipedia.py: Demonstrates the use of online VB for LDA to
# analyze a bunch of random Wikipedia articles.
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cPickle, string, numpy, getopt, sys, random, time, re, pprint, gc, os

import onlineldavb
from lxml import html

# import file for easy access to browser database
sys.path.append('BasicBrowser/')
import db

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
    
def main():
    """
    Analyzes scraped pages using scikit-learn.LDA
    """
    
    # The number of topics
    K = 10
    # no of documents
    D = 300
    n_features = 1000

    # Our vocabulary
    vocab = list(set(file('./vocab').readlines()))
    W = len(vocab)
    
    # Add terms and topics to the DB
    db.init()
    db.add_terms(vocab)
    db.add_topics(K)
    
    olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)

    # grab documents
    ### Load your scraped pages, re-tokenize, and vectorize result.
    docset, docnames = [], []
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
                docset += [clean_html(rawtext)]
                docnames += [filename[:-5]]
                if not(len(docset) % 10): print("loaded " + str(len(docset)) + " documents")

    # Give them to online LDA
    # Also computes an estimate of held-out perplexity
    (wordids, wordcts) = onlineldavb.parse_doc_list(docset, olda._vocab)
    (gamma, bound) = olda.update_lambda(wordids, wordcts)

    
    # Arrays for adding batches of data to the DB
    # doc_array = []
    # doc_term_array = []

    # for d in range(len(docnames)):
        # doc_array.append((docnames[d], docset[d]))
    doc_array = zip(docnames, docset)
        
    # Add a batch of docs to the DB; this is the one DB task that is not in
    # the separate DB write thread since later tasks depend on having doc ids.
    # Since writes take so long, this also balaces the two threads time-wise.
    doc_ids = db.add_docs(doc_array)

    doc_topic_array = []
    for d in range(len(gamma)):
        doc_size = len(docset[d])
        for k in range(len(gamma[d])):
            doc_topic_array.append((doc_ids[d], k, gamma[d][k], gamma[d][k]/doc_size))
    db.add_doc_topics(doc_topic_array)

    perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))
    print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
        (1, olda._rhot, numpy.exp(-perwordbound))

    # Save lambda, the parameters to the variational distributions
    # over topics, and gamma, the parameters to the variational
    # distributions over topic weights for the articles analyzed in
    # the last iteration.
    numpy.savetxt('lambda-%d.dat' % 1, olda._lambda)
    numpy.savetxt('gamma-%d.dat' % 1, gamma)
        
    topic_terms_array = []
    for topic in range(len(olda._lambda)):
        lambda_sum = sum(olda._lambda[topic])
            
        for term in range(len(olda._lambda[topic])):
            topic_terms_array.append((topic, term, olda._lambda[topic][term]/lambda_sum))
    db.update_topic_terms(K, topic_terms_array)
            
    gc.collect() # probably not necesary, but precautionary for long runs
    db.print_task_update()

    # The DB thread ends only when it has both run out of tasks and it has been
    # signaled that it will not be recieving any more tasks
    db.increment_batch_count()
    db.signal_end()

if __name__ == '__main__':
    main()
