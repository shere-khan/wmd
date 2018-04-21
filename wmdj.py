import os, re, pickle

import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from pyemd import emd
from scipy.spatial import distance

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

# def wmd(X1, X2, docs1, docs2):
#     X1 = normalize(X1, norm='l1', copy=False)
#     X2 = normalize(X2, norm='l1', copy=False)
#     distances = euclidean_distances(docs1, docs2)
    # res = emd(X1, X2, distances)

    # return res

def remove_emoji_and_nums(text):
    emojis = """:-) :) :o) :] :3 :c) :> =] 8) =) :} :^) 
    :D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 B^D :( :/ :-( :'( :D :P""".split()
    emojipat = "|".join(map(re.escape, emojis))
    text = re.sub("[^a-zA-Z0-9{0}]".format(emojipat), " ", text)

    return text

def cleantext(text):
    pass

def normlz(doc):
    tot = 0
    for x in doc:
        tot += x
    return [x/tot for x in doc]
#
def build_w2v_feature(nBOW, doc, bow_m, w2v_m, feat_size):
    vocab = set(w2v_m.wv.vocab)
    w2vlist = np.zeros((len(nBOW), feat_size))
    for w in doc:
        if w in vocab:
            v = bow_m.vocabulary_
            i = v.get(w.lower())
            if i is not None:
                w2vlist[i] = w2v_model[w]

    return w2vlist


if __name__ == '__main__':
    # EMD test
    # first_histogram = np.array([0.0, 1.0])
    # second_histogram = np.array([5.0, 3.0])
    # distance_matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
    # emd(first_histogram, second_histogram, distance_matrix)

    # euc dist example
    # D = np.array([[1, 3], [1, 2]])
    # D_prime = np.array([[2, 2], [1, 1]])
    # res = euclidean_distances(D, D_prime)

    data  = util.extract_raw_data(['/home/justin/pycharmprojects/rnn_sent_analysis_6640'
                                 '/data/processed/'], cap=None)

    # Train BOW
    # train = [x[0].split("+:::")[0] for x in data]
    # vectorizer = CountVectorizer(analyzer="word", max_features=1000)
    # vectorizer.fit(train)
    # pickle.dump(vectorizer, open("data/vectorizer", "wb"))

    bow_model = pickle.load(open("data/vectorizer", "rb"))
    w2v_model = Word2Vec.load("/home/justin/pycharmprojects/rnn_sent_analysis_6640/data"
                              "/w2v_3000")
    train = [x[0].split("+:::")[0] for x in data]

    doc1 = train[0]
    doc2 = train[1]

    vocab = set(w2v_model.wv.vocab)
    doc1 = [x for x in doc1.split() if x in vocab]
    doc2 = [x for x in doc2.split() if x in vocab]

    # transform to BOW
    X1 = bow_model.transform([" ".join(doc1)])
    X2 = bow_model.transform([" ".join(doc2)])

    # nX1 = normalize(X1, norm='l1', copy=False)
    # nX2 = normalize(X2, norm='l1', copy=False)

    # Normalize
    nX1 = np.array(normlz(X1.toarray().tolist()[0]))
    nX2 = np.array(normlz(X2.toarray().tolist()[0]))

    # Create W2V features for distance matrix
    w2v_X1 = build_w2v_feature(nX1, doc1, bow_model, w2v_model, 3000)
    w2v_X2 = build_w2v_feature(nX2, doc2, bow_model, w2v_model, 3000)

    # w2v_X1 = [w2v_model[x] for x in doc1]
    # w2v_X2 = [w2v_model[x] for x in doc2]

    bow_vocab = bow_model.vocabulary_
    for key, val in bow_vocab.items():
        if val == 126:
            print(key)
        if val == 30:
            print(key)

    v1 = w2v_model["but"]
    v2 = w2v_model["all"]
    res1 = distance.euclidean(v1, v2)

    # Calculate dists between w2v feature lists
    W_dist = euclidean_distances(w2v_X1, w2v_X2)

    res = emd(nX1, nX2, W_dist)
    # res = emd.emd.emd(nX1, nX2)
    print(res)

