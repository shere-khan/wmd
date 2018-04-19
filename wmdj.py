import os, re, pickle

import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from pyemd import emd
from gensim.models import Word2Vec
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cosine
from sklearn.metrics import euclidean_distances
from gensim.models import KeyedVectors
# from pyemd import emd
# from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import normalize
from sentiment import util

stops = set(stopwords.words('english'))

def create_vocab_dict():
    if not os.path.exists("data/embed.dat"):
        print("Caching word embeddings in memmapped format...")
        wv = KeyedVectors.load_word2vec_format(
            "data/GoogleNews-vectors-negative300.bin.gz", binary=True)
        wv.init_sims()
        fp = np.memmap("data/embed.dat", dtype=np.double, mode='w+',
                       shape=wv.syn0norm.shape)
        fp[:] = wv.syn0norm[:]
        with open("data/embed.vocab", "w") as f:
            for _, w in sorted((voc.index, word) for word, voc in wv.vocab.items()):
                print(w, file=f)
        del fp, wv

    with open("data/embed.vocab") as f:
        vocab_list = map(str.strip, f.readlines())

    return {w: k for k, w in enumerate(vocab_list)}

def wmd(X1, X2, docs1, docs2):
    X1 = normalize(X1, norm='l1', copy=False)
    X2 = normalize(X2, norm='l1', copy=False)
    distances = euclidean_distances(docs1, docs2)
    res = emd(X1, X2, distances)

    return res

def remove_emoji_and_nums(text):
    emojis = """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^) 
    :D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 B^D :( :/ :-( :'( :D :P""".split()
    emojipat = "|".join(map(re.escape, emojis))
    text = re.sub("[^a-zA-Z0-9{0}]".format(emojipat), " ", text)

    return text

def cleantext(text):
    pass

if __name__ == '__main__':
    # EMD test
    # first_histogram = np.array([0.0, 1.0])
    # second_histogram = np.array([5.0, 3.0])
    # distance_matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
    # emd(first_histogram, second_histogram, distance_matrix)

    data  = util.extract_raw_data(['/home/justin/pycharmprojects/rnn_sent_analysis_6640'
                                 '/data/processed/'], cap=None)

    # Train BOW
    # train = [x[0].split("+:::")[0] for x in data]
    # vectorizer = CountVectorizer(analyzer="word", max_features=1000)
    # vectorizer.fit(train)
    # pickle.dump(vectorizer, open("data/vectorizer", "wb"))

    bow_model = pickle.load("data/vectorizer")
    w2v_model = Word2Vec.load("/home/justin/pycharmprojects/rnn_sent_analysis_6640/data"
                              "/w2v_nn")
    train = [x[0].split("+:::")[0] for x in data]

    doc1 = train[0]
    doc2 = train[1]

    X1 = bow_model.transform(doc1)
    X2 = bow_model.transform(doc2)
    n_X1 = normalize(X1, norm='l1', copy=False)
    n_X2 = normalize(X2, norm='l1', copy=False)

    w2v_X1 = [w2v_model[x] for x in doc1]
    w2v_X2 = [w2v_model[x] for x in doc2]
    W_dist = euclidean_distances(doc1, doc2)

    res = emd(X1, X2, W_dist)

