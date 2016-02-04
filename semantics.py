from db_manager import *
import word2vec

SQL_DB = 'sqlite:///clean_chronicling_america.sqlite'
WORDS_FILE = 'docs'
PHRASES_FILE = 'docs-phrases'
BIN_FILE = 'docs.bin'
CLUSTER_FILE = 'docs-clusters.txt'


def dump_text():
    session = connect(SQL_DB)
    r = session.execute('select ocr_eng from articles')

    with open(WORDS_FILE, 'w') as f:
        for i, doc in enumerate(r):
            f.write(doc[0] + '\n')

if __name__=='__main__':
    # word2vec.word2phrase(WORDS_FILE, PHRASES_FILE, verbose=True)
    # word2vec.word2vec(PHRASES_FILE, BIN_FILE, size=100, verbose=True)
    # word2vec.word2clusters(WORDS_FILE, CLUSTER_FILE, 100, verbose=True)
    # model = word2vec.load(BIN_FILE)
    # print(model.vocab)
    # print(model.vectors.shape)
    clusters = word2vec.load_clusters(CLUSTER_FILE)
    print(clusters)