import logging
import itertools

import numpy as np
import gensim
import re

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

WIKI = '/Users/benjaminbolte/Desktop/topics/topic_modeling_tutorial/data/simplewiki-20150603-pages-articles.xml.bz2'


def head(stream, n=10):
    return list(itertools.islice(stream, n))


from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS


def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS and re.match("^[A-Za-z0-9_-]*$", token)]


def iter_wiki(dump_file, n=-1):
    ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    counter = 0
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        counter += 1
        if counter == n:
            break
        text = filter_wiki(text)
        tokens = tokenize(text)
        if len(tokens) < 50 or any(title.startswith(ns+':') for ns in ignore_namespaces):
            continue
        yield title, tokens

id2words = {0: u'word', 2: u'profit', 300: u'another_word'}
doc_stream = (tokens for _, tokens in iter_wiki(WIKI, 1000))
id2word_wiki = gensim.corpora.Dictionary(doc_stream)
id2word_wiki.filter_extremes(no_below=20, no_above=0.1)

print(id2word_wiki)