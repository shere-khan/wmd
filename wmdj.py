import os

import numpy as np
from pyemd import emd
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

if __name__ == '__main__':
    # EMD test
    first_histogram = np.array([0.0, 1.0])
    second_histogram = np.array([5.0, 3.0])
    distance_matrix = np.array([[0.0, 0.5], [0.5, 0.0]])
    emd(first_histogram, second_histogram, distance_matrix)

    # Create lexicon
    lexicon = create_vocab_dict()

    # Create mem-map to access small chunks of the
    W = np.memmap("data/embed.dat", dtype=np.double, mode="r", shape=(3000000, 300))

    # Create training and test datasets out of 20NewsGroup data
    newsgroups = fetch_20newsgroups()
    docs, y = newsgroups.data, newsgroups.target
    docs_train, docs_test, y_train, y_test = train_test_split(docs, y, train_size=100,
                                                              test_size=300,
                                                              random_state=0)
    # Fit BOW model out of 20NewsGroup dataset
    vect = CountVectorizer(stop_words="english").fit(docs_train + docs_test)

    # Get common words between lexicon and new BOW model vocab
    common = [word for word in vect.get_feature_names() if word in lexicon]
    model = W[[lexicon[w] for w in common]]

    # Make new BOW model comprising only of words in common
    vect = CountVectorizer(vocabulary=common, dtype=np.double)

    # Create vector representations out of training and test documents
    X_train = vect.fit_transform(docs_train)
    X_test = vect.transform(docs_test)

    doc1 = X_train[0]
    doc2 = X_train[1]

    dist1 = [model[x] for x in docs_train[0]]
    dist2 = [model[x] for x in docs_train[1]]
    wmd(doc1, doc2, dist1, dist2)

