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
from sklearn.metrics.pairwise import cosine_similarity
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

def write_bow_vocab(vocab):
    with open("data/bow_vocab.txt", "w") as f:
        for i, v in enumerate(list(vocab)):
            f.write("{0:<15}".format(v))
            print("{0:<15}".format(v), end="")
            if i % 8 == 0:
                f.write("\n")
                print()

def get_similar_words(words, model, allwords):
    for w in words.split():
        print("{0:>15}".format(w))
        if w in model:
            sim = w2v_model.wv.most_similar(w)
            for s in sim:
                if s[0] in allwords:
                    print("{0:>15} {1:<.5f}".format(s[0], s[1]))
            print()
        else:
            print("NA")

def all_words_present(doc, allwords):
    for d in doc.split():
        if d not in allwords:
            print("{0} not present".format(d))

def wmd(doc1, doc2, bow1, bow2, featsize, bow_model, w2v_model):
    # Normalize
    nX1 = np.array(normlz(bow1.toarray().tolist()[0]))
    nX2 = np.array(normlz(bow2.toarray().tolist()[0]))

    # Create W2V features for distance matrix
    w2v_X1 = build_w2v_feature(nX1, doc1, bow_model, w2v_model, featsize)
    w2v_X2 = build_w2v_feature(nX2, doc2, bow_model, w2v_model, featsize)

    # Calculate dists between w2v feature lists
    W_dist = euclidean_distances(w2v_X1, w2v_X2)

    res = emd(nX1, nX2, W_dist)
    print(res)

def train_bow():
    train = [x[0].split("+:::")[0] for x in data]
    vectorizer = CountVectorizer(analyzer="word", max_features=5000)
    vectorizer.fit(train)
    pickle.dump(vectorizer, open("data/vectorizer", "wb"))
    exit(0)

if __name__ == '__main__':
    data  = util.extract_raw_data(['/home/justin/pycharmprojects/rnn_sent_analysis_6640'
                                 '/data/processed/'], cap=None)

    # train_bow()

    bow_model = pickle.load(open("data/vectorizer", "rb"))
    w2v_model = Word2Vec.load("/home/justin/pycharmprojects/rnn_sent_analysis_6640/data"
                              "/w2v_3000")
    bow_vocab = bow_model.vocabulary_
    w2v_vocab = set(w2v_model.wv.vocab)
    allwords = list(set(bow_vocab.keys()).intersection(w2v_vocab))
    # write_bow_vocab(allwords)
    doc1 = "the pretentious squad receives negative review"
    doc2 = "the pretentious squad receives negative comment"
    # doc2 = "the pretentious squad receives a negative comment"
    # doc2 = "insipid unit discovers another positive comment"
    # doc2 = "something completely different"
    all_words_present(doc1, allwords)
    all_words_present(doc2, allwords)
    # get_similar_words(doc1, w2v_model, set(allwords))
    # get_similar_words(doc2, w2v_model)
    # exit(0)

    train = [x[0].split("+:::")[0] for x in data]

    # doc1 = train[0]
    # doc2 = train[1]

    doc1 = [x for x in doc1.split() if x in w2v_vocab]
    doc2 = [x for x in doc2.split() if x in w2v_vocab]

    # transform to BOW
    bow1 = bow_model.transform([" ".join(doc1)])
    bow2 = bow_model.transform([" ".join(doc2)])

    wmd(doc1, doc2, bow1, bow2, 3000, bow_model, w2v_model)
    cs = cosine_similarity(bow1, bow2)
    # cs = cosine_similarity(np.array([0, 1]), np.array([1, 0]))
    print("cosine-similarity: {0:0.3f}".format(cs[0][0]))
