import logging
import itertools

import numpy as np
import gensim
import re

import nltk
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

from gensim.utils import lemmatize, tokenize
from gensim.parsing.preprocessing import STOPWORDS

from db_manager import *
import datetime

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

SQL_DB = 'sqlite:///clean_chronicling_america.sqlite'
DICT_NAME = 'cca.dict'


def simple_tokenize(text):
    return [token for token in tokenize(text, lower=True) if token not in STOPWORDS and len(token) > 3]


class DocumentTokenizer(object):
    def __init__(self, doc_it):
        self.doc_it = doc_it
        words = itertools.chain.from_iterable(self.split_words(text) for _, _, text in doc_it)
        self.bigrams, self.trigrams = self.best_ngrams(words)

    def tokenize(self, message):
        text = u' '.join(self.split_words(message))
        text = re.sub(self.trigrams, lambda match: match.group(0).replace(u' ', u'_'), text)
        text = re.sub(self.bigrams, lambda match: match.group(0).replace(u' ', u'_'), text)
        return text.split()

    @staticmethod
    def split_words(text):
        return [token for token in gensim.utils.tokenize(text, lower=True) if token not in STOPWORDS and len(token) > 3]

    @staticmethod
    def best_ngrams(words, top_n=1000, min_freq=100):
        tcf = TrigramCollocationFinder.from_words(words)
        tcf.apply_freq_filter(min_freq)
        trigrams = [' '.join(w) for w in tcf.nbest(TrigramAssocMeasures.chi_sq, top_n)]
        logging.info('%i trigrams found: %s...' % (len(trigrams), trigrams[:10]))

        bcf = tcf.bigram_finder()
        bcf.apply_freq_filter(min_freq)
        bigrams = [' '.join(w) for w in bcf.nbest(BigramAssocMeasures.pmi, top_n)]
        logging.info('%i bigrams found: %s...' % (len(bigrams), bigrams[:10]))

        pat_gram2 = re.compile('(%s)' % '|'.join(bigrams), re.UNICODE)
        pat_gram3 = re.compile('(%s)' % '|'.join(trigrams), re.UNICODE)

        return pat_gram2, pat_gram3


class DocumentIterator(object):
    def __init__(self, db, limit=-1, date_range=[1800, 2015]):
        self.db = db
        self.limit = limit
        self.date_range = date_range

    def __iter__(self):
        session = connect(self.db)
        r = session.execute("select * from articles where date > %i and date < %i limit %i" % (self.date_range[0] * 10000, self.date_range[1] * 10000, self.limit))

        for doc in r:
            text = doc['ocr_eng']
            date = str(doc['date'])
            state = doc['state']
            date = datetime.date(day=int(date[6:]), month=int(date[4:6]), year=int(date[:4]))
            yield date, state, text


class DocumentBowIterator(object):
    def __init__(self, doc_it, doc_wiki):
        self.doc_it = doc_it
        self.doc_wiki = doc_wiki
        self.states = set()
        self.min_date = datetime.date.today()
        self.max_date = datetime.date(day=1, month=1, year=1800)

    def __iter__(self):
        for date, state, text in self.doc_it:
            self.states.add(state)

            if date < self.min_date:
                self.min_date = date
            if date > self.max_date:
                self.max_date = date

            yield date, state, self.doc_wiki.doc2bow(simple_tokenize(text))


def similarity_metric(wf_one, wf_two):
    s_one = {val: cnt for val, cnt in wf_one}
    s_two = {val: cnt for val, cnt in wf_two}
    denom = sum(cnt for val, cnt in itertools.chain.from_iterable((wf_one, wf_two)))
    if denom < 20: return 0
    return float(sum(s_one[val] + cnt for val, cnt in s_two.iteritems() if s_one.has_key(val))) / denom


if __name__=='__main__':
    try:
        doc_wiki = gensim.corpora.Dictionary.load(DICT_NAME)
    except IOError:
        doc_wiki = gensim.corpora.Dictionary(simple_tokenize(text) for _, _, text in DocumentIterator(SQL_DB))
        doc_wiki.filter_extremes(no_below=30, no_above=0.1)
        doc_wiki.save(DICT_NAME)

    docs = [(date, state, vector) for date, state, vector in DocumentBowIterator(DocumentIterator(SQL_DB, date_range=[1917, 1918]), doc_wiki)]

    try:
        similarity_metric = np.loadtxt('SIM_MAT.mat', delimiter=',')
    except IOError:
        similarity_metric = np.array([[0 if v1 == v2 else similarity_metric(v1, v2) for _, _, v2 in docs] for _, _, v1 in docs])
        np.savetxt('SIM_MAT.mat', similarity_metric, delimiter=',')

    state_labels = [doc[1] for doc in docs]
    unique_states = list(set(state_labels))
    state_label_map = { unique_states[i]: i for i in range(len(unique_states)) }

    state_hist = np.array([state_label_map[l] for l in state_labels])
    state_individ = np.array([state_label_map[l] for l in unique_states])
    state_counts = np.bincount(state_hist)

    state_similarity_metric = np.array([[similarity_metric[np.ix_(state_hist == state_one, state_hist == state_two)].mean() for state_two in state_individ] for state_one in state_individ])

    np.savetxt('1917_similarity_matrix.txt', state_similarity_metric, delimiter=',')
