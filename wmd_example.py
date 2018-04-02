import os

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import cosine
from sklearn.metrics import euclidean_distances
from gensim.models import KeyedVectors
# from pyemd import emd
# from gensim.models.word2vec import Word2Vec


if __name__ == '__main__':
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

    W = np.memmap("data/embed.dat", dtype=np.double, mode="r", shape=(3000000, 300))
    with open("data/embed.vocab") as f:
        vocab_list = map(str.strip, f.readlines())

    vocab_dict = {w: k for k, w in enumerate(vocab_list)}

    d1 = "Obama speaks to the media in Illinois"
    d2 = "The President addresses the press in Chicago"

    vect = CountVectorizer(stop_words="english").fit([d1, d2])
    print("Features:", ", ".join(vect.get_feature_names()))

    v_1, v_2 = vect.transform([d1, d2])
    v_1 = v_1.toarray().ravel()
    v_2 = v_2.toarray().ravel()
    print(v_1, v_2)
    print("cosine(doc_1, doc_2) = {:.2f}".format(cosine(v_1, v_2)))

    W_ = W[[vocab_dict[w] for w in vect.get_feature_names()]]
    D_ = euclidean_distances(W_)
    print("d(addresses, speaks) = {:.2f}".format(D_[0, 7]))
    print("d(addresses, chicago) = {:.2f}".format(D_[0, 1]))
