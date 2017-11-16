

from __future__ import unicode_literals, division
import json
import time
from io import open
import unicodedata
import string
import re
import random
import pickle

SOS_token = 0
EOS_token = 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z0-9.!?]+", r" ", s)
    return s

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def clean_data(paths):
    l = Lang('squad')
    for test_path in paths:
        with open(test_path) as data_file:
            dataset = json.load(data_file)
        data_file.close()
        dataset = dataset['data']
        for article_idx, article in enumerate(dataset):
            for paragraph_idx, paragraph in enumerate(article['paragraphs']):
                passage = normalizeString(paragraph['context'])
                l.addSentence(passage)
                for qa in paragraph['qas']:
                    qa_id = qa['id']
                    question = normalizeString(qa['question'])
                    l.addSentence(question)
    print 'vocab size: ', l.n_words
    '''
    with open('../data/word2index.json', 'w') as f:
        f.write(unicode(json.dumps(l.word2index, ensure_ascii=False)))
    f.close()
    with open('../data/index2word.json', 'w') as f:
        f.write(unicode(json.dumps(l.index2word, ensure_ascii=False)))
    f.close()
    '''
    return l

if __name__ == '__main__':
    t1 = time.time()

    paths = ['../data/testing.json', \
    '../data/development.json', \
    '../data/training.json']
    l = clean_data(paths)
    t2 = time.time() - t1
    print 'collect data with out write: ', t2
    output = open('../data/lang.pkl', 'wb')
    pickle.dump(l, output)

    #favorite_color = pickle.load( open( "../data/lang.pkl", "rb" ) )

    #print favorite_color.n_words
    print 'collect data time: ', time.time() - t1
