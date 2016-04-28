# babywiki

## Introduction 
``Alan'' is an experimental package to help automatically seed and structure a topic- or query-specific knowledge-base---or, in less jargon---to automatically draft baby (topic- or query-specific) Wikipedias. 

``Alan'' uses a focused web-crawler to collect documents on the web, filtering for relevance based on content similarity and/or domain and link information, organizes the crawled pages using a LDA topic model, and helps the user to visualize the topic model using a Topic Model Visualization browser as envisioned by Allison Chaney (see https://github.com/ajbc/tmv).

## Setup 
This package is written in Python. The various libraries and dependencies it needs are listed in requirements.txt. If you use  pip an easy way to get them is to use 

pip install -r requirements.txt

More specifically, the crawler only requires NLTK and Scrapy, and the topic modeling (which is integrated with the topic model visualization) is what requires NumPy and Django. Do note that the topic model visualization specifically requires an older version of Django (1.2.4); newer versions of Django have incompatible APIs.

You may need to manually download certain NLTK corpora (mainly the stopwords corpus and the corpus for the Punkt tokenizer); for this you may use the NLTK Downloader, by opening a Python shell and typing >>> nltk.download()

## Crawling
Once you have all the dependencies, you may crawl a collection of documents by going to the package directory (i.e. where README.md is located) and typing

scrapy crawl veblen -a query="your query here" -o topvect.csv

to generate a topic term frequency vector based on your query, and then

scrapy crawl kythera -a query="your query here" -o scores.csv

Depending on your system and network setup, and with the default crawler settings, the crawl may take anywhere between 10 minutes to a couple (or up to 4 or so) hours to run, and may use a gigabyte or two of memory. 

If you find that the crawler becomes too resource-intensive, consider using a job directory (see / Google the Scrapy documentation for more details on this.)

## Topic modeling
You can then run ldabuild.bat (on Windows machines) or ldabuild.sh (on *NIX systems) to pass the collection of documents into a LDA topic model; this process will also set up and build the database used by the topic model visualization browser. 

## Topic model visualization
Finally, to run the topic model visualization browser, go to the BasicBrowser directory and execute

python manage.py runserver

and then navigate your browser to 127.0.0.1:8080/topic\_presence.

You can also run ldanmf.py to see topic models (without the interactive visualization.) This requires scikit-learn.
