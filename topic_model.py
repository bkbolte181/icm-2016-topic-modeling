import logging
import os
import sys
import re
import tarfile
import itertools

import nltk
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

import gensim
from gensim.parsing.preprocessing import STOPWORDS

from textblob import TextBlob

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

TAR_FILE = '../topic_modeling_tutorial/data/20news-bydate.tar.gz'


def open_tar(tar_name):
    """ Open a tar file """
    with tarfile.open(tar_name, 'r:gz') as tf:
        file_infos = [file_info for file_info in tf if file_info.isfile()]
        return tf.extractfile(file_infos[0]).read()


def process_message(message):
    message = gensim.utils.to_unicode(message, 'latin1').strip()
    blocks = message.split(u'\n\n')
    content = u'\n\n'.join(blocks[1:-1])
    return content


def iter_20newsgroups(file_name, log_every=None):
    extracted = 0
    with tarfile.open(file_name, 'r:gz') as tf:
        for file_number, file_info in enumerate(tf):
            if file_info.isfile():
                if log_every and extracted % log_every == 0:
                    logging.info("extracting 20newsgroups file #%i: %s" % (extracted, file_info.name))
                content = tf.extractfile(file_info).read()
                yield process_message(content)
                extracted += 1


class Corpus20News(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        for text in iter_20newsgroups(self.file_name):
            yield list(gensim.utils.tokenize(text, lower=True))


class Corpus20News_Lemmatize(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        for message in iter_20newsgroups(self.file_name):
            yield self.tokenize(message)

    def tokenize(self, text):
        return [t for t in gensim.utils.lemmatize(text) if t.split('/')[0] not in STOPWORDS]


def best_ngrams(words, top_n=1000, min_freq=100):
    tcf = TrigramCollocationFinder.from_words(words)
    tcf.apply_freq_filter(min_freq)
    trigrams = [' '.join(w) for w in tcf.nbest(TrigramAssocMeasures.chi_sq, top_n)]
    logging.info('%i trigrams found: %s...' % (len(trigrams), trigrams[:20]))

    bcf = tcf.bigram_finder()
    bcf.apply_freq_filter(min_freq)
    bigrams = [' '.join(w) for w in bcf.nbest(BigramAssocMeasures.pmi, top_n)]
    logging.info('%i bigrams found: %s...' % (len(bigrams), bigrams[:20]))

    pat_gram2 = re.compile('(%s)' % '|'.join(bigrams), re.UNICODE)
    pat_gram3 = re.compile('(%s)' % '|'.join(trigrams), re.UNICODE)

    return pat_gram2, pat_gram3


class Corpus20News_Collocations(object):
    def __init__(self, file_name):
        self.file_name = file_name
        logging.info('collecting ngrams from %s' % self.file_name)
        documents = (self.split_words(text) for text in iter_20newsgroups(self.file_name, log_every=1000))
        words = itertools.chain.from_iterable(documents)
        self.bigrams, self.trigrams = best_ngrams(words)

    def split_words(self, text, stopwords=STOPWORDS):
        return [word for word in gensim.utils.tokenize(text, lower=True) if word not in STOPWORDS and len(word) > 3]

    def tokenize(self, message):
        text = u' '.join(self.split_words(message))
        text = re.sub(self.trigrams, lambda match: match.group(0).replace(u' ', u'_'), text)
        text = re.sub(self.bigrams, lambda match: match.group(0).replace(u' ', u'_'), text)
        return text.split()

    def __iter__(self):
        for message in iter_20newsgroups(self.file_name):
            yield self.tokenize(message)


def head(stream, n=10):
    return list(itertools.islice(stream, n))


def best_phrases(document_stream, top_n=1000, prune_at=50000):
    np_counts = {}
    for docno, doc in enumerate(document_stream):
        if docno % 1000 == 0:
            sorted_phrases = sorted(np_counts.iteritems(), key=lambda  item: -item[1])
            np_counts = dict(sorted_phrases[:prune_at])
            logging.info('at document #%i, considering %i phrases: %s...' % (docno, len(np_counts), head(sorted_phrases)))

        for np in TextBlob(doc).noun_phrases:
            if u' ' not in np:
                continue

            if all(word.isalpha() and len(word) > 2 for word in np.split()):
                np_counts[np] = np_counts.get(np, 0) + 1

    sorted_phrases = sorted(np_counts, key=lambda np: -np_counts[np])
    return set(head(sorted_phrases, top_n))


class Corpus20News_NE(object):
    def __init__(self, file_name):
        self.file_name = file_name
        logging.info('collecting entities from %s' % self.file_name)
        doc_stream = itertools.islice(iter_20newsgroups(self.file_name), 10000)
        self.entities = best_phrases(doc_stream)
        logging.info('selected %i entities: %s...' % (len(self.entities), list(self.entities)[:10]))

    def __iter__(self):
        for message in iter_20newsgroups(self.file_name):
            yield self.tokenize(message)

    def tokenize(self, message, stopwords=STOPWORDS):
        result = []
        for np in TextBlob(message).noun_phrases:
            if u' ' in np and np not in self.entities:
                continue
            token = u'_'.join(part for part in gensim.utils.tokenize(np) if len(part) > 2)
            if len(token) < 4 or token in stopwords:
                continue
            result.append(token)
        return result

ne_corpus = Corpus20News_NE(TAR_FILE)
print(head(ne_corpus, 5))