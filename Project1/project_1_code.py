# -*- coding: utf-8 -*-
"""
Project 1: Language Modeling and Word Embeddings
Python version: 2.x
Group Member: Mengyao Lyu (ml2559), Ruoyang Sun (rs2385), Qianhao Yu (qy99)
"""


# library 
import nltk
import pandas as pd
import numpy as np


################################### Functions ###################################
"""
filepath(string)： file path
return:
l (list/string): tokens
"""
def read_dev(filepath):
    l = []
    with open(filepath, 'r') as f:
        for line in f:
            tokens = nltk.word_tokenize(line.lower())
            l.append(tokens)
    return l

"""
path_name(string): input path 

return:
uni_dict, bi_dict (dict): count of unigram, bigram (respectively)
uni_df, bi_df (DataFrame): count of unigram, bigram (respectively) in DataFrame
"""
def preprocess(path_name):

    # input path 
    path = nltk.data.find(path_name)
    raw = open(path, 'rU').read()

    # break raw text into tokens 
    tokens = nltk.word_tokenize(raw)
    remove_tokens = ['\'', '\`','\``']
    tokens = [x for x in tokens if x not in remove_tokens]

    # prepare 
    len_tokens = len(tokens)
    end_tokens = ['.', '...', '?', '!', ';']
    uni_dict = {}
    bi_dict = {}
    uni_dict['<\start>'] = 1 
    pre = '<\start>'
    new_len = len_tokens + 1

    # count unigram
    for i in range(len_tokens):

        word = tokens[i].lower()
        if word in uni_dict:
            uni_dict[word] += 1
        else:
            uni_dict[word] = 1
        if word in end_tokens:
            uni_dict['<\start>'] += 1

    # count bigram, add unknown
    for i in range(len_tokens):

        word = tokens[i].lower()
        if uni_dict[word] == 1:
            word = '<UNK>'

        # bigram count
        if pre in end_tokens: 
            uni_dict['<\start>'] += 1
            bi_combine = pre + " " + '<\start>'

            if bi_combine in bi_dict:
                bi_dict[bi_combine] += 1
            else:
                bi_dict[bi_combine] = 1

            new_len += 1
            pre = '<\start>'
            bi_combine = pre + " " + word

            if bi_combine in bi_dict: 
                bi_dict[bi_combine] += 1
            else: 
                bi_dict[bi_combine] = 1

        else:
            bi_combine = pre + " " + word
            if bi_combine in bi_dict:
                    bi_dict[bi_combine] += 1
            else:
                bi_dict[bi_combine] = 1
        pre = word

    # add unknown in unigram
    uni_dict['<UNK>'] = 0
    for key in uni_dict.keys():
        if uni_dict[key] == 1:
            uni_dict['<UNK>'] += 1
            uni_dict.pop(key, None)

            """ create dataframe to store count """

    # unigram count table
    uni_df = pd.DataFrame(uni_dict.items(), columns=['Tokens', 'Count'])
    
    uni_df = uni_df.set_index('Tokens')
    uni_df.index.name = None

    # bigram count table 
    len_verc = len(uni_dict)
    prob2 = np.zeros((len_verc, len_verc))
    bi_df = pd.DataFrame(prob2, index = uni_dict.keys(), columns = uni_dict.keys())
    
    # 
    for k, v in bi_dict.items():
        spl = k.split(" ")
        bi_df.loc[spl[0], spl[1]] = v #* 1.0 / verc[spl[0]]

    return uni_dict, bi_dict, uni_df, bi_df


def uni_prob(count_df_uni):

    prob_uni_df = count_df_uni.copy()
    len_t = np.sum(count_df_uni.loc[:, 'Count'].values)
    prob_uni_df.Count /= len_t * 1.0

    return prob_uni_df

"""
uni_dict, bi_cnt_df (DataFrame): DataFrame of count of unigram, bigram (respectively)
k (float): smoothing parameter

return:
bi_prob_df (DataFrame): Dataframe of probability of bigram
"""

def smooth(uni_dict, bi_cnt_df, k = 0.005):

    v = len(uni_dict)

    # if bigram, convert to DataFrame and add k to each count in the table 
    bi_prob_df = bi_cnt_df.copy()
    tmp = bi_prob_df.loc[:, :].values
    tmp += k
    bi_prob_df.loc[:, :] = tmp

    for key in bi_prob_df.columns:
        bi_prob_df.loc[key, :] /=  uni_dict[key] * 1.0 + v * k

    return bi_prob_df

"""
prob_df_uni, prob_df_bi (DataFrame): DataFrame of probability of unigram, bigram (respectively)
list_of_words (list/string): a sentence
hyper_l (float): interpolate parameter lambda
mode(int): 1 or 0
threshold (float): cut-off value
"""
def interpolate_perplexity(prob_df_uni, prob_df_bi, list_of_words, hyper_l, mode, threshold):

    keys = prob_df_bi.columns.values
    unknown = '<UNK>'
    pre = '<\start>'
    res = 0
    for word in list_of_words:
        if word in keys:
            prob_uni = prob_df_uni.loc[word, :].values
            prob_bi = prob_df_bi.loc[pre, word]
            pre = word
        else:
            prob_uni = prob_df_uni.loc[unknown, :].values
            prob_bi = prob_df_bi.loc[pre, unknown]
            pre = unknown

        if mode == 0 or prob_bi > threshold:
            prob = prob_uni * hyper_l + (1 - hyper_l) * prob_bi
        else:
            prob = prob_uni
        res -= np.log(prob) 

    return res[0], len(list_of_words)

"""
Calculate the perplexity of the whole text (corpse)

prob_df_uni, prob_df_bi (DataFrame)
l_pos (list)
"""
def perplexity(prob_df_uni, prob_df_bi, l_pos):

    pp_uni = 0.0
    pp_bi = 0.0
    cnt1 = 0
    cnt2 = 0

    for line in l_pos:
        pp1, n1 = interpolate_perplexity(prob_df_uni, prob_df_bi, line, hyper_l = 1, mode = 0, threshold = 0.00000001)
        pp2, n2 = interpolate_perplexity(prob_df_uni, prob_df_bi, line, hyper_l = 0, mode = 0, threshold = 0.00000001)
        pp_uni += pp1
        pp_bi += pp2
        cnt1 += n1
        cnt2 += n2
    pp_uni = np.exp(pp_uni/ cnt1)
    pp_bi = np.exp(pp_bi/ cnt2)

    return pp_uni, pp_bi


"""
Make sentiment classification based:
input: test data set, trained model on positive and negative corpora, using Unigram or not, method = "perplexity" or "embedding"
output: list of prediction (pos: 0, neg:1)
"""
def classify(test_set,prob_df_uni_pos, prob_df_bi_pos,prob_df_uni_neg, prob_df_bi_neg, method = 'perplexity'):

    sentences = read_dev(test_set)
    result = []

    if method == 'perplexity': 
        for line in sentences:
            pos_pp, _ = interpolate_perplexity(prob_df_uni_pos, prob_df_bi_pos, line, hyper_l=0, mode = 1, threshold = 1e-8)
            neg_pp, _ = interpolate_perplexity(prob_df_uni_neg, prob_df_bi_neg, line, hyper_l=0, mode = 1, threshold = 1e-8)
            if pos_pp < neg_pp:
                result.append(0)
            else:
                result.append(1)
        return result
    else:
        return NotImplemented


"""
return prediction accuracy given lists of predictions and true labels
"""
def accuracy(pred, true_label):

    compare = lambda x,y: x==y
    acc = [compare(x,y) for x in pred for y in true_label]

    return 100.0 * sum(acc)/len(acc)


def testOneK(k, line_list, count_df_uni_pos, count_df_uni_neg, count_df_bi_pos, count_df_bi_neg, uni_dict_pos, uni_dict_neg, hyper_l, mode, threshold):

    res = np.zeros((len(line_list), 1))
    i = 0
    prob_df_uni_pos = uni_prob(count_df_uni_pos)
    prob_df_uni_neg = uni_prob(count_df_uni_neg)
    prob_df_bi_pos = smooth(uni_dict_pos, count_df_bi_pos, k)
    prob_df_bi_neg = smooth(uni_dict_neg, count_df_bi_neg, k)

    for list_of_words in line_list:
        pp_pos, tmp1 = interpolate_perplexity(prob_df_uni_pos, prob_df_bi_pos, list_of_words, hyper_l, mode, threshold)
        pp_neg, tmp1 = interpolate_perplexity(prob_df_uni_neg, prob_df_bi_neg, list_of_words, hyper_l, mode, threshold)
        if pp_pos < pp_neg:
            res[i, 0] = 1
        else:
            res[i, 0] = 0
        i += 1

    return np.sum(res)

"""
return: a matrix with k as columns and lambda as rows, each cell is filled with corresponding accuracy * corpse length
"""
def chooseK(k_list, lm_list, l_pos, l_neg, count_df_uni_pos, count_df_uni_neg, count_df_bi_pos, count_df_bi_neg, uni_dict_pos, uni_dict_neg, mode=1, thre=1e-8):

    acc = np.zeros((len(k_list), len(lm_list)))
    neg_len = len(l_neg)
    
    for i in range(len(k_list)):
        for j in range(len(lm_list)): 
            res_pos = testOneK(k_list[i], l_pos, count_df_uni_pos, count_df_uni_neg, count_df_bi_pos, count_df_bi_neg, uni_dict_pos, uni_dict_neg, lm_list[j], mode, thre)
            res_neg = testOneK(k_list[i], l_neg, count_df_uni_pos, count_df_uni_neg, count_df_bi_pos, count_df_bi_neg, uni_dict_pos, uni_dict_neg, lm_list[j], mode, thre)
            acc[i, j] = (res_pos +  neg_len - res_neg) #* 1.0 / total_len
    idx1 = np.argmax(acc)
    jj_lm = idx1 % len(lm_list)
    ii_k = idx1 / len(lm_list)

    return k_list[ii_k], lm_list[jj_lm], acc

def saveResult(result):
    res = np.array(result).reshape(len(result), 1)
    idx = np.arange(1, len(result) +1,dtype='int32').reshape(len(result), 1)
    newres = np.hstack((idx, res))
    newres = np.int32(newres)
    df3 = pd.DataFrame(newres, columns = ['Id', 'Prediction'])
    df3.to_csv('./submission_12.csv',index = False)
    print newres.shape, newres[:, 0], type(newres[0, 0])

def main():
    path_pos = '/Users/mlyu/Documents/Cornell/fa17/cs5740/homework/proj1/Project1/SentimentDataset/Train/pos.txt'
    path_neg = '/Users/mlyu/Documents/Cornell/fa17/cs5740/homework/proj1/Project1/SentimentDataset/Train/neg.txt'

    path_dev_pos = './SentimentDataset/Dev/pos.txt'
    path_dev_neg = './SentimentDataset/Dev/neg.txt'

    path_test = '/Users/mlyu/Documents/Cornell/fa17/cs5740/homework/proj1/Project1/SentimentDataset/Test/test.txt'

    print("load data ... Calculate n-gram...")
    uni_dict_pos, bi_dict_pos, uni_df_pos, bi_df_pos = preprocess(path_pos)
    uni_dict_neg, bi_dict_neg, uni_df_neg, bi_df_neg = preprocess(path_neg)

    l_pos = read_dev(path_dev_pos)
    l_neg = read_dev(path_dev_neg)
    #test_set = read_dev(path_test)

    # find the optimal k and lambda
    '''
    N = 4
    K = 3
    k_list = [0.001, 0.002, 0.003, 0.004, 0.005]
    lm_list = 1.0/K * np.arange(K)
    k, l, acc = chooseK(k_list, lm_list, l_pos, l_neg, uni_df_pos, uni_df_neg, bi_df_pos, bi_df_neg, uni_dict_pos, uni_dict_neg, mode=0, thre=1e-8)
    print(k, l, acc)
    '''
    # part 5
    print("smooth... get probability...")
    prob_df_uni_pos = uni_prob(uni_df_pos)
    prob_df_bi_pos = smooth(uni_dict_pos, bi_df_pos,k=0.003)
    prob_df_uni_neg = uni_prob(uni_df_neg)
    prob_df_bi_neg = smooth(uni_dict_neg, bi_df_neg,k=0.003)
    print("make prediction...")
    result = classify(path_test, prob_df_uni_pos, prob_df_bi_pos,prob_df_uni_neg, prob_df_bi_neg, method = 'perplexity')
    saveResult(result)

if __name__ == '__main__':
    main()
