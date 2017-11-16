
# coding: utf-8

# In[90]:

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gensim
import math
import json
from pprint import pprint
import nltk
import io
import unicodedata
import string
import re
import random

import time
from pycorenlp import StanfordCoreNLP

import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer # tfidf
import numpy as np
from scipy import spatial
import pandas as pd
from nltk import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[92]:

with open("../data/stopwords.txt", "r") as stopwords:
    remove_token = stopwords.readlines()[0].split(",") + list(string.punctuation)
stopwords.close()

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
glove_dict = load_glove('/Users/ruoyangsun/Desktop/Fall2017/cs5740/Project1/part2/glove.6B/glove.6B.300d.txt')


# In[93]:

def glove2vec(token):
    if token not in glove_dict:
        return np.ones((1, 300))
    return glove_dict[token].reshape((1,300))


# In[102]:

def findSentence(passage, question, k):
    '''
    input: whole paragraph text, current quetion

    output: key sentence's index
    '''

    sentences = nltk.tokenize.sent_tokenize(passage)
    vectorizer_tf = TfidfVectorizer()
    m = vectorizer_tf.fit_transform(sentences)
    q = vectorizer_tf.transform([question])
    sim = cosine_similarity(q,m)
    k = min(k, len(sentences))    
    sentence_idx = sim.flatten().argsort()[-k:]
    try:
        sent =  [ sentences[i] for i in sentence_idx ]
    except Exception as error:
        print('TypeError: string indices must be integers')
        sent = [sentences[0]]
    return (sent, sentence_idx)

def findAnswer(tokens_json, ner_pos, targets):
    '''
    helper function for chooseByType
    collect all words which fit our requirements
    '''
    
    ans = ''
    for token in tokens_json:
        if token[ner_pos] in targets:
            ans += ' ' + token['word']
    return ans[1:]


def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
#    return tokens
    filtered_words = [w for w in tokens if not w in remove_token]
    return filtered_words

def big_k(m_list, k):
    table = pd.DataFrame(m_list,columns=["value"])
    table["idx"] = pd.DataFrame(np.arange(len(m_list)))
    table = table.sort_values(by="value",ascending=False)
    return table.idx[:k].tolist()


def most_similar(cand, target, k):
    """
    find most similar k sentences based on coseline similarities on average of word embeddings
    """
    # cut cand paragraph in to sentence in a list
    cand = tokenize.sent_tokenize(cand)

    # only keep keyword for both cand and target
    cand_keyword = [preprocess(x) for x in cand]
    #target_keyword = preprocess(target)
    target_keyword = target
    try:
        target_keyword_vec = glove2vec(target_keyword[0])

    except:
        target_keyword_vec = np.ones((1, 300))

        #print('1st unknown word in preprocess!')
    cnt = 1
    for i in range(1, len(target_keyword)):
        try:
            target_keyword_vec = np.add(target_keyword_vec, glove2vec(target_keyword[i]))
            cnt +=1
        except:
            #print('unknown word in preprocess!')
            kk = 0
    target_keyword_vec = np.divide(target_keyword_vec, cnt)

    if k > len(cand): k = len(cand)

    m_cos = []

    for tokens in cand_keyword:

        # vectorize
        try:
            cur_words = glove2vec(tokens[0])

        except:
            cur_words = np.ones((1, 300))
            #print('1st unknown word in preprocess!')
        cnt = 1
        for i in range(1, len(tokens)):
            try:
                cur_words = np.add(cur_words, glove2vec(tokens[i]))
                cnt += 1
            except:
                #print('unknown word in preprocess!')
                kb = 0
        cur_words = np.divide(cur_words, cnt)

        m_cos.append(spatial.distance.cosine(target_keyword_vec, cur_words))
        idx = big_k(m_cos, k)
        res = [cand[i] for i in idx]

    return (res, idx)

def candidate_answer(sents, answer_type, data):
    candidate_answer = {}
    potential_answer = {}
    sentences = sents[0]
    idx = sents[1]
    for t, sent in enumerate(sentences):
        sent_idx = idx[t]
        sent = sent.encode('ascii','ignore')
        print('idx t:', sent_idx)

        #nlp_tokens = getCoreNLP(article_idx,paragraph_idx, nlp)['sentences'][0]['tokens']
        try:
            nlp_tokens = data['sentences'][sent_idx]['tokens']
        except:
            sent_idx = 0
            nlp_tokens = data['sentences'][sent_idx]['tokens']

        i = 0
        j = 0
        n = len(nlp_tokens)
        while i < n and j < n:
            target = nlp_tokens[i]
            j = i + 1

            if target['ner'] in answer_type and target['pos'] == 'NNP':
                if target['word'] not in candidate_answer:
                    candidate_answer[target['word']] = [t * 100 + target['index']]
                else:
                    candidate_answer[target['word']].append(t * 100 + target['index'])

                phrase = [target['word']]
                while  j < n and nlp_tokens[j]['ner'] in answer_type and nlp_tokens[j]['pos'] == 'NNP':
                    phrase.append(nlp_tokens[j]['word'])
                    if nlp_tokens[j]['word'] not in candidate_answer:
                        candidate_answer[nlp_tokens[j]['word']] = [t * 100 + nlp_tokens[j]['index']]
                    else:
                        candidate_answer[nlp_tokens[j]['word']].append(t * 100 + nlp_tokens[j]['index'])
                    j += 1

                if len(phrase) == 1:
                    break
                nnp = " ".join(phrase)

                if nnp not in candidate_answer:
                    candidate_answer[nnp] = [sent_idx * 100 + target['index']]
                else:
                    candidate_answer[nnp].append(sent_idx * 100 + target['index'])
                i = j
                continue

            elif target['ner'] in answer_type:
                if target['word'] not in candidate_answer:
                    candidate_answer[target['word']] = [sent_idx * 100 + target['index']]
                else:
                    candidate_answer[target['word']].append(sent_idx * 100 + target['index'])
            else:
                if target['pos'] == 'NNP':
                    if target['word'] not in potential_answer:
                        potential_answer[target['word']] = [sent_idx * 100 + target['index']]
                    else:
                        potential_answer[target['word']].append(sent_idx * 100 + target['index'])
            i += 1

    if len(candidate_answer) == 0:
        return potential_answer
    return candidate_answer

def questionProcess(question):
    '''
    choose rules:
        c who/whom/ --> ner: PERSON
        c when/what time --> ner: DATE, TIME, DURATION
        c what/which --> pos: NN, NNS, NNP, NNPS
        why/how --> pos: VB, VBD, VBG, VBN, VBP, VBZ
        c how many/ how much --> ner: NUMBER, MONEY
        c where --> ner: LOCATION, ORGANIZATION
    input: data['sentences'][0]['tokens'], question
    '''

    who = ['PERSON']
    when_what_time = ['DATE', 'TIME', 'DURATION']
    where = ['LOCATION', 'ORGANIZATION']
    how_many_how_much = ['NUMBER', 'MONEY']

    question_tokens = nltk.word_tokenize(question)
    qlist  = ["who","whom", "whose", "what", "where", "when", "why", "how", "which"]

    flag = 0
    for idx, word in enumerate(question_tokens):
        if word.lower() in qlist:
            flag = 1
            question_word = word.lower()
            q_idx = idx
            if q_idx >= len(question_tokens) - 2:
                q_words = question_tokens[:q_idx]
            else:
                q_words = question_tokens[q_idx + 1:]
            break
    
    if flag == 0:
        q_idx = 0
        q_words = question_tokens
        question_word = 'what'

    ans_type = "MISC"
    if question_word == 'where':
        ans_type = where
    elif question_word in ['who','whose','whom']:
        ans_type = who
    elif question_word == 'when' or (question_word == 'what' and question_tokens[q_idx + 1] == 'time'):
        ans_type = when_what_time
    elif question_word == 'how':
        if question_tokens[q_idx + 1] in ['much','many','little','few']:
            ans_type = 'QUANTITY'
        elif question_tokens[q_idx + 1] in ['young','old']:
            ans_type = when_what_time
    else:
        ans_type = "MISC"
    
    filtered_words = [w for w in q_words if not w in remove_token]
    return ans_type, filtered_words

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

def getCoreNLP(article_idx,paragraph_idx,coreNLP):
    '''helper function to read json output from CoreNLP'''

    passage_idx = str(article_idx) + " " + str(paragraph_idx)
    return coreNLP[passage_idx]

def load_json(test_path):
    with open(test_path) as data_file:
        dataset = json.load(data_file)
    data_file.close()
    return dataset

def write_json(result, out_path):
    with io.open(out_path,'w',encoding="utf-8") as outfile:
        outfile.write(unicode(json.dumps(result, ensure_ascii=False)))
    outfile.close()

def KNN_Pre_process(ques, cands, dim = 300):
    can_num = len(cands.keys())
    Y = np.zeros((can_num, dim))
    Y_dict = {}
    vec2 = np.ones((1, dim))
    for i, can in enumerate(cands.keys()):
        Y_dict[i] = can
        word_l = can.split(' ')
        vec = np.zeros((1, dim))
        cnt = 0
        for token in word_l:
            cnt += 1
            try:
                vec += np.asarray(glove2vec(token))
            except:
                print('unknow words in google glove2vec')
                vec += vec2
        Y[i, :] = vec / cnt

    X = np.zeros((len(ques), dim))
    for i, token in enumerate(ques):
        try:
            vec = np.asarray(glove2vec(token))
        except:
            print('unknown word in question key words!')
            vec = np.zeros((1, dim))
        X[i, :] = vec
    return X, Y, Y_dict


def KNN(X, Y, k, Y_dict, cands):
    if X.shape[0] == 0 or Y.shape[0] == 0:
        return 'they'
    Y[np.isnan(Y)] = np.median(Y[~np.isnan(Y)])
    X[np.isnan(X)] = np.median(X[~np.isnan(X)])
    Z = cosine_similarity(X, Y)
    #print(Z)
    k = min(k, Y.shape[0])
    k_span_inv = np.argsort(Z, axis=1)[:, -k:]
    #print(k_span_inv)
    #k_span = np.flip(k_span_inv, axis = 1)
    idx_rank = {}
    for i in range(X.shape[0]):
        for j in range(k):
            idx = k_span_inv[i, j]
            if idx in idx_rank:
                idx_rank[idx] += np.exp(j)
            else:
                idx_rank[idx] = np.exp(j)
    sort_idx_rank = sorted(idx_rank.iteritems(), key = lambda (k,v):(v,k), reverse=True)
    ans = {}
    #print(sort_idx_rank)
    #print(Y_dict)
    for tu in sort_idx_rank[:k]:
        token = Y_dict[tu[0]]
        posi = cands[token][0]
        ans[posi] = token
    #print(ans)
    tmp = sorted(ans.iteritems())
    key = ''
    for tu in tmp:
        key += tu[1] + ' '
    return key[: -1]


def chooseByType(tokens_json, question):
    '''
    choose rules: 
        c who/whom/ --> ner: PERSON
        c when/what time --> ner: DATE, TIME, DURATION
        c what/which --> pos: NN, NNS, NNP, NNPS
        why/how --> pos: VB, VBD, VBG, VBN, VBP, VBZ
        c how many/ how much --> ner: NUMBER, MONEY
        c where --> ner: LOCATION, ORGANIZATION
    input: data['sentences'][0]['tokens'], question
    '''
    
    who_whom = ['PERSON']
    when_what_time = ['DATE', 'TIME', 'DURATION']
    what_which = ['NN', 'NNS', 'NNP', 'NNPS']
    why = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    how_many_how_much = ['NUMBER', 'MONEY']
    where = ['LOCATION', 'ORGANIZATION']
    
    question_tokens = nltk.word_tokenize(question)
    start = question_tokens[0].lower()
    second = question_tokens[1].lower()
    flag = 0
    if start == 'where':
        flag = 1
        answer = findAnswer(tokens_json, 'ner', where)
    elif start == 'who' or start == 'whom':
        flag = 1
        answer = findAnswer(tokens_json, 'ner', who_whom)
    elif start == 'when' or ((start == 'what' or start == 'which') and (second == 'time' or second == 'year')):
        flag = 1                     
        answer = findAnswer(tokens_json, 'ner', when_what_time)
    elif (start == 'how' and second == 'many') or (start == 'how' and second == 'much'):
        flag = 1
        answer = findAnswer(tokens_json, 'ner', how_many_how_much)
    elif start == 'why' or start == 'how':
        answer = findAnswer(tokens_json, 'pos', why)
    elif start == 'what' or start == 'which':
        answer = findAnswer(tokens_json, 'pos', what_which)
    else:
        answer = findAnswer(tokens_json, 'pos', what_which)
    return answer

def chooseCand(sents, question, data):
    candidate_answer = {}
    potential_answer = {}
    sentences = sents[0]
    idx = sents[1]
    ans = ''
    
    for t, sent in enumerate(sentences):
        sentence_idx = idx[t]
        sent = sent.encode('ascii','ignore')
        #print('idx t:', sentence_idx)

        try:
            if sentence_idx >= len(data['sentences']):
                sentence_idx = len(data['sentences']) - 1
        except Exception as error:
            print('TypeError: string indices must be integers')
            break
        tokens_json = data['sentences'][sentence_idx]['tokens']
        an = chooseByType(tokens_json, question)
        ans += an + ' '
        
    ans_dict = {}
    for i, a in enumerate(ans.split(' ')):
        if a in question.split(' '):
            continue
        if a in ans_dict:
            l = ans_dict[a]
            l.append(i)
        else:
            ans_dict[a] = [i]
    return ans_dict

    
def QA_TFIDF(dataset, nlp, k = 5):
    result = {}
    dataset = dataset['data']
    em = 0 
    match = 0
    cnt = 0
    for article_idx, article in enumerate(dataset):
        for paragraph_idx, paragraph in enumerate(article['paragraphs']):
            passage = paragraph['context'].encode('ascii','ignore')
            data = getCoreNLP(article_idx, paragraph_idx, nlp)
            
            if article_idx == 0 and paragraph_idx == 1:
                print(passage)
            
            for qa in paragraph['qas']:
                qa_id = qa['id']
                question = qa['question']
                sentences = findSentence(passage, question, 1) #（sentences, idx_list）
                candidate = chooseCand(sentences, question, data)
                
                filtered_words = [w for w in question.split(' ')[-5:] if not w in remove_token]
                X, Y, Y_dict = KNN_Pre_process(filtered_words, candidate, dim = 300)
                ans = KNN(X, Y, k, Y_dict, candidate)
                #ans = " ".join(candidate.keys())
                result[qa_id] = ans              
                cnt += 1
                real_ans = qa['answers'][0]['text'].split(' ')
                
                if article_idx == 0 and paragraph_idx == 1:  
                    print('question: ', question)
                    print('Real Answer: ', real_ans)
                    print('Top Sentence: ', sentences)
                    print('candidate_answer: ', candidate)
                    print('Our answer: ' , ans)
                ii = 0
                ans_list = candidate.keys()
                for real_an in real_ans:
                    if real_an in ans_list:
                        ii += 1
                if ii == len(real_ans):
                    em += 1
                if ii > 0:
                    match += 1

    print('candidate answer matching percentage: ', match * 100.0 / cnt, '%')
    print('candidate answer exactly matching percentage: ', em * 100.0 / cnt, '%')            
    return result

def main():
    dev_path = '../data/development.json'
    test_path = '../data/testing.json'
    dev_coreNLP = '../data/dev_coreNLP.json'
    test_coreNLP = '../data/test_coreNLP.json'
    dev_dataset = load_json(dev_path)
    dev_tag = load_json(dev_coreNLP)
    test_dataset = load_json(test_path)
    test_tag = load_json(test_coreNLP)
    dev_result = QA_TFIDF(dev_dataset, dev_tag)
    test_result = QA_TFIDF(test_dataset, test_tag)

    dev_out_path = '../results/dev_glove_tfidf.json'
    test_out_path = '../results/test_glove_tfidf.json'

    write_json(dev_result, dev_out_path)
    write_json(test_result, test_out_path)
main()

