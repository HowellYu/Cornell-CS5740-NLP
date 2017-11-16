# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
import os
import random
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
# torch.cuda.set_device(args.gpu)
import torch.utils.data as Data
import pickle
import json

from io import open
import unicodedata
import string
import re
import random

import sys
import numpy as np
import subprocess

from torch.autograd import Variable
from torch.nn import Parameter

from load_word import Lang
from PointerNet import PointerNet


class ParaEnc(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(ParaEnc, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        #print 'D size(): ', lstm_out.size()
        return lstm_out[:, 0, :].t() #(seq_len, batch, hidden_size * num_directions)

class QueEnc(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(QueEnc, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        linear_out = self.linear(lstm_out)
        tanh_out = self.tanh(linear_out)
        #print 'Q size(): ', tanh_out.size()
        return tanh_out[:, 0, :].t()

class Encoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):#embedding_dim, hidden_dim, vocab_size, model_dir):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.para = ParaEnc(embedding_dim, hidden_dim, vocab_size)
        self.que = QueEnc(embedding_dim, hidden_dim, vocab_size)

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, para, que):
        D = self.para(para) #(50, 124)
        Q = self.que(que) #(50, 10)
        L = D.t().mm(Q) # (124, 10)
        Aq = F.softmax(L) # (124, 10)
        Ad = F.softmax(L.t()) # (10, 124)
        Cq = D.mm(Aq) # (50, 10)
        Cd = torch.cat((Q, Cq), 0).mm(Ad) #100,124
        #print 'Cd size: ', Cd.size() #
        return Cd

'''
lstm input: input, (h_0, c_0)
input (seq_len, batch, input_size)
h_0 (num_layers * num_directions, batch, hidden_size)
c_0 (num_layers * num_directions, batch, hidden_size)
'''
class PtrNet(nn.Module):

    def __init__(self, hidden_dim):
        super(PtrNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.zeros_2l = Variable(torch.zeros(2 * hidden_dim, 1))
        self.linear1 = nn.Linear(2 * hidden_dim, hidden_dim, False) # VH
        self.linear2 = nn.Linear(hidden_dim, hidden_dim) #Wh+b
        self.linear3 = nn.Linear(hidden_dim, 1, False)
        #self.hidden = self.init_hidden()
        self.c = Parameter(torch.Tensor(1))
        self.lstm = nn.LSTM(2 * hidden_dim, hidden_dim)


    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim), requires_grad = False), \
            Variable(torch.zeros(1, 1, self.hidden_dim), requires_grad = False))
    '''
    H size: 2l * p
    '''
    def forward(self, H):
        seq_length = H.size(1)
        probs = Variable(torch.Tensor(1, seq_length + 1))
        H_r_2d = torch.cat((H, self.zeros_2l), 1) # 2l * p+1
        H_r = torch.cat((H, self.zeros_2l), 1).t().contiguous().view(seq_length + 1, 1, -1) # p+1 *1*2l
        #x = H_r.continuous().view(-1, 1, 2 * self.hidden_dim)
        VH = self.linear1(H_r) # p+1 * 1 * l
        hidden = self.init_hidden() #h0, c0

        for _ in range(seq_length):
            x2 = self.linear2(hidden[0]) # 1 * 1 * l
            n = x2.expand(seq_length + 1, 1, self.hidden_dim) # p+1 * 1 * l
            f = F.tanh(VH + n) # (p + 1) * 1 * l
            vf = self.linear3(f).view(-1, 1) # (p+1 ) * 1 * 1 (P+1) * 1
            beta = F.softmax((vf + self.c).t()) # 1 * p+1
            #print(beta)
            #x = vf.permute(1, 2, 0).bmm(H_r.permute(1, 0, 2)) #If batch1 is a b x n x m Tensor, batch2 is a b x m x p Tensor,
            # 1 * 1 * 2l
            x = H_r_2d.mm(beta.t()).view(1, 1, -1) #2l * 1
            o, (h, c) = self.lstm(x, hidden) #todo
            hidden = (h, c)
            #values, indices = beta.max(0)
            probs = torch.cat((probs, beta), 0)
        out = F.log_softmax(probs)
        return out

class SQUAD(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(SQUAD, self).__init__()
        self.encode = Encoder(embedding_dim, hidden_dim, vocab_size)
        #self.decode = Decoder(2 * hidden_dim, vocab_size)
        self.ptr = PtrNet(hidden_dim)
        self.init_hidden()

    def forward(self, para, que):
        Cd = self.encode(para, que)
        probs = self.ptr(Cd)
        return probs

    def init_hidden(self):
        self.encode.init_hidden()
        self.ptr.init_hidden()



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

def word2idx(lang, sentence):
    l = []
    for sent in sentence.split(' '):
        l.append(lang.word2index[sent])
    ll = Variable(torch.LongTensor(l), requires_grad=False)
    return ll

def formatAns(start_idx, ans, length_p):
    l = range(start_idx, start_idx + len(ans.split(' ')))
    l.append(length_p)
    y = Variable(torch.LongTensor(l), requires_grad= False)
    return y, len(l)

def get_true_start_idx(paragraph, start_idx, ans, span = 5):
    pl = paragraph.split(' ')
    l = len(paragraph[:start_idx].split(' '))
    ansl = ans.split(' ')
    if start_idx == 0:
        return 0
    elif l < span:
        #print('less than span')
        return l - 2
    else:
        upper = min(len(pl), l + span)
        for i in range(l - span, upper):
            if ansl[0] == pl[i]:
                return i
        print('not found')
        return l - 2
def train_epoch(dataset, model, lang, loss_func, optimizer, learning_rate = 1e-3):
    avg_loss = 0.0
    cnt = 0
    truth_res = []
    pred_res = []
    batch_sent = []

    for article_idx, article in enumerate(dataset):
        for paragraph_idx, paragraph in enumerate(article['paragraphs']):
            passage = normalizeString(paragraph['context'])
            v_p = word2idx(lang, passage)
            length_p = len(passage.split(' '))
            for qa in paragraph['qas']:
                cnt += 1
                qa_id = qa['id']
                question = normalizeString(qa['question'])
                v_q = word2idx(lang, question)
                start_idx = qa['answers'][0]['answer_start']
                ans = normalizeString(qa['answers'][0]['text'])
                true_start_idx = get_true_start_idx(passage, start_idx, ans)
                y, length = formatAns(true_start_idx, ans, length_p)

                model.init_hidden()
                model.zero_grad()
                probs = model.forward(v_p, v_q)
                y_pred = probs[:length, :]
                loss = loss_func(y_pred, y)
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
                optimizer.step()
                avg_loss += loss.data[0]
                if cnt % 10 == 0:
                    print('iterations loss: ', cnt, loss.data[0])
                    for param in model.parameters():
                        print(param.grad, param.size())
                if cnt % 100 == 0:
                    print('iterations avg loss', cnt, avg_loss)
    print('train epoch is over')
    print('average loss is: ', ava_loss / cnt)
    return model

def idx2word(ans, lang, thre = 0.9):
    array = ans.data.numpy()
    idx = np.where(array > thre)[1]
    res = ''
    if len(idx) == 0:
        return lang.index2word[np.random.randint(len(array))]
    else:
        for i in idx:
            res += lang.index2word[i] + ' '
        return res

def evaluate(model, test_path, lang):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = {}
    print('start evaluation...')
    dataset = load_json(test_path)
    i = 0
    for article_idx, article in enumerate(dataset):
        i += 1
        for paragraph_idx, paragraph in enumerate(article['paragraphs']):
            passage = normalizeString(paragraph['context'])
            v_p = word2idx(lang, passage)
            for qa in paragraph['qas']:
                qa_id = qa['id']
                question = normalizeString(qa['question'])
                v_q = word2idx(lang, question)
                model.init_hidden()
                ans = model(v_p, v_q)
                res = idx2word(ans, lang)
                pred_res[qa_id] = res
        print('finished %i article: ' %(i))
    print('start writing result to disk...')
    output_path = '../results/test_res_2.json'
    with open(output_path, 'w') as outfile:
        json.dump(test_result, outfile)
    outfile.close()
    subprocess.call('python ./evaluate.py ' + test_path + ' ' + output_path, shell=True)


def embedOne(ans, vocab_size, lang):
    ll = 0.0 * Variable(torch.FloatTensor(1, vocab_size), requires_grad=False)
    for sent in ans.split(' '):
        ll[0, lang.word2index[sent]] = 1.0
    return ll


def load_json(test_path):
    with open(test_path) as data_file:
        dataset = json.load(data_file)
    data_file.close()
    return dataset['data']

def train():
    embedding_dim = 50
    hidden_dim = 50
    pkl_path = '../data/lang.pkl'
    lang = pickle.load(open(pkl_path, "rb"))
    vocab_size = lang.n_words
    #dev_path = '../data/development.json'
    train_path = '../data/training.json'
    dataset = load_json(train_path)
    model = SQUAD(embedding_dim, hidden_dim, vocab_size)
    loss_func = torch.nn.NLLLoss()
    learning_rate = 1e-3
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-3)
    print('start training...')
    epoch = 10
    for i in range(epoch):
        model = train_epoch(dataset, model, lang, loss_func, optimizer, learning_rate)
        test_path = '../data/testing.json'
        evaluate(model, test_path, lang)



if __name__ == '__main__':
    dev_path = '../data/development.json'
    dev_tags_path = '../data/dev_coreNLP.json'
    train()
    '''
    pkl_path = '../data/lang.pkl'
    lang = pickle.load(open(pkl_path, "rb"))
    ans = Variable(torch.rand(1, lang.n_words), requires_grad = False)
    l = idx2word(ans, lang, thre = 0.99)
    print l
    '''








