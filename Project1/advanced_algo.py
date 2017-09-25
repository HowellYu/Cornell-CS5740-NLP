import nltk
import pandas as pd
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from gensim import corpora, models, similarities
import gensim
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

path_train_pos = "./SentimentDataset/Train/pos.txt"
path_train_neg = "./SentimentDataset/Train/neg.txt"
path_dev_pos = './SentimentDataset/Dev/pos.txt'
path_dev_neg = './SentimentDataset/Dev/neg.txt'
path_test = './SentimentDataset/Test/test.txt'

def read_corpus(filepath):
    l = []
    with open(filepath) as f:
        for line in f:
            l.append(line)
    return l

def getY(corpus_train_pos, corpus_train_neg):
    n1 = len(corpus_train_pos)
    n2 = len(corpus_train_neg)
    Y1 = np.zeros((n1, 1))
    Y2 = np.ones((n2, 1))
    y_train = np.vstack((Y1, Y2))
    y_train = y_train.reshape(y_train.shape[0],)
    return y_train

corpus_train_pos = read_corpus(path_train_pos)
corpus_train_neg = read_corpus(path_train_neg)
corpus_dev_pos = read_corpus(path_dev_pos)
corpus_dev_neg = read_corpus(path_dev_neg)
corpus_pre = read_corpus(path_test)
corpus_train = corpus_train_pos + corpus_train_neg
corpus_val = corpus_dev_pos + corpus_dev_neg
corpus_test = read_corpus(path_test)


#==================
## TF-IDF
#==================#
vectorizer_tf = TfidfVectorizer()
X_train = vectorizer_tf.fit_transform(corpus_train)
X_val = vectorizer_tf.transform(corpus_val)
X_test = vectorizer_tf.transform(corpus_test)

y_train = getY(corpus_train_pos, corpus_train_neg)
y_val = getY(corpus_dev_pos, corpus_dev_neg)

def classify_sklearn(clf, X_train, y_train, X_val, y_val):
    clf.fit(X_train, y_train)
    y_pre = clf.predict(X_val)
    acc = len(np.where(y_pre == y_val.T)[0]) * 1.0 / len(y_pre)
    print acc
    return clf,y_pre

def ensembleVote(*args):
    result = np.zeros(args[0].size)
    for arg in args:
        result = np.add(result, arg)
    result = np.divide(result, len(args))
    result[result >= 0.5] = 1
    result[result <0.5] = 0
    return result

# NN
clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12, 10), random_state=1)
clf1,y_nn = classify_sklearn(clf1, X_train, y_train, X_val, y_val)

def getResult(clf, X_test):
    result = clf.predict(X_test)
    res = np.array(result).reshape(len(result), 1)
    idx = np.arange(1, len(result) +1,dtype='int32').reshape(len(result), 1)
    newres = np.hstack((idx, res))
    newres = np.int32(newres)
    df3 = pd.DataFrame(newres, columns = ['Id', 'Prediction'])
    df3.to_csv('./submission_NB.csv',index = False)
    print newres.shape, newres[:, 0], type(newres[0, 0])

# SVM
scaler = StandardScaler()
X = scaler.fit_transform(X_train.toarray())
X2 = scaler.fit_transform(X_val.toarray())
clf_svm = svm.SVC()
clf_svm,y_svm = classify_sklearn(clf_svm, X_train, y_train, X_val, y_val)

## Naive Bayes
a_list = [0.71,0.73, 0.75,0.77, 0.79]
for i in a_list:
    clfNB = MultinomialNB(alpha = i)
    classify_sklearn(clfNB, X_train, y_train, X_val, y_val)

clfNB = MultinomialNB(alpha = 0.71)
clbNB= classify_sklearn(clfNB, X_train, y_train, X_val, y_val)
getResult(clfNB, X_test)


#==================
## word2vec
# ================#
word2vec_corpus = './word2vec/GoogleNews-vectors-negative300.bin'
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_corpus, binary=True)

def word_embedding2(tokens, dic):
    vector = np.zeros(300)
    length = 0
    for word in tokens:
        if word not in dic:
            continue
        vector = np.add(vector, dic.wv[word])
        length+=1
    result = np.divide(vector,length)
    return result

X_train = np.array(map(lambda x: word_embedding2(x,word2vec_model),corpus_train))
X_val = np.array(map(lambda x: word_embedding2(x,word2vec_model),corpus_val))

# NN
clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 12), random_state=1)
clf2 = classify_sklearn(clf2, X_train, y_train, X_val, y_val)

## SVM
clf_svm = svm.SVC(kernel = 'linear')
clf_svm,y_svm = classify_sklearn(clf_svm, X_train, y_train, X_val, y_val)

## NB
clfNB = GaussianNB()
clfNB , y_nb = classify_sklearn(clfNB, X_train, y_train, X_val, y_val)

#==================
## glove
# ================#
def load_glove(file_path):
    vocab = {} #skip information on first line
    file= open(file_path)    
    for line in file:
        items = line.split(' ')
        word = items[0]
        vect = np.array([float(i) for i in items[1:] if len(i) >1])
        if len(vect) !=300:
            continue
        vocab[word] = vect
    return vocab

def word_embedding(tokens, dic):
    vector = np.zeros(300)
    length = 0
    for word in tokens:
        if word not in dic:
            continue
        vector = np.add(vector, dic[word])
        length+=1
    result = np.divide(vector,length)
    return result

glove_dict = load_glove('./glove.6B/glove.6B.300d.txt')

X_train = np.array(map(lambda x: word_embedding(x,glove_dict),corpus_train))
X_val = np.array(map(lambda x: word_embedding(x,glove_dict),corpus_val))

# NN
clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 12), random_state=1)
clf1 = classify_sklearn(clf1, X_train, y_train, X_val, y_val)

## SVM
clf_svm = svm.SVC(kernel = 'linear')
clf_svm,y_svm = classify_sklearn(clf_svm, X_train, y_train, X_val, y_val)

## NB
clfNB = GaussianNB()
clfNB = classify_sklearn(clfNB, X_train, y_train, X_val, y_val)