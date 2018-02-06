import numpy as np
from sets import Set
import pandas as pd
from collections import defaultdict
import time

from NGram_tri import ngram

class HMM(ngram):
	def __init__(self, train_path, test_path):
		ngram.__init__(self, train_path)
		self.test_path = test_path
		self.tags_list = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']
		self.val_list = None
		self.recalulate_bigram = True

	def getNgram(self):
		self.first_pass()
		self.sec_pass()
		self.get_transition()
		self.get_emission()
		self.get_transition_uni()
		# self.cnt2prob()
		self.get_word2idx(k1 = 1.0, k2 = 1.0)
		print 'ngram is ready.'


	def viterbi(self,lambda1=0.0,lambda2=0.0,lambda3=1.0):
		#the Viterbi algorithm
		'''
		input:
		interpolation coefficients

		output:
		
		'''
		f = open(self.test_path, "r")
		start_idx = []
		total_score = np.zeros((9,0))
		total_bp = np.zeros((9,0))
		cnt = 0
		result = []
		i = 0
		for line in f:
			words = line.split()
			pos = f.next().split()
			index = f.next().split()
			length = len(words)
			score = np.zeros((9,length))
			bp = np.zeros((9,length))
			cnt += length
			pred_tags = self.viterbi_singleLine(words, pos, l1 = lambda1,l2 = lambda2,l3= lambda3)
			result = result + pred_tags
	    
		f.close()
		return result

	def process(self, predict_list):

	    storage = {'PER':[], 'LOC':[], 'ORG':[], 'MISC':[]}

	    i = 0
	    length = len(predict_list)
	    while(i < length):
	        start_index = i
	        end_idx = i
	        if predict_list[i][:2] == 'B-':
	            while (i + 1 < length and predict_list[i][2:] == predict_list[i+1][2:]):
	                i += 1
	            end_idx = i
	            storage[predict_list[start_index][2:]].append(str(start_index) + "-" + str(end_idx))
	        # if there are more than 1 I with the same type, we consider this be a prediction
	        '''
	        elif predict_list[i][:2] == 'I-':
	        	while (i + 1 < length and predict_list[i][2:] == predict_list[i+1][2:]):
	        		i += 1
	        	end_idx = i
	        	if end_idx > start_index:
	        		storage[predict_list[start_index][2:]].append(str(start_index) + "-" + str(end_idx))
	        '''
	        i += 1
	    return storage

	def out2csv(self, result, output):
	    per = result['PER']
	    loc = result['LOC']
	    org = result['ORG']
	    misc = result['MISC']

	    org_str = " ".join(org)
	    misc_str = " ".join(misc)
	    per_str = " ".join(per)
	    loc_str = " ".join(loc)
	    
	    d = {'Type': ['ORG','MISC','PER','LOC'] , 'Prediction': [org_str,misc_str,per_str,loc_str]}
	    df = pd.DataFrame(d, columns = d.keys())
	    df.to_csv(output,index = False)

	def train(self, l1, l2, l3):
		print 'start training...'
		if self.recalulate_bigram:
			self.getNgram()
		print 'start viterbi...'
		predict_list = self.viterbi(l1, l2, l3)
		result = self.process(predict_list)
		print 'prediction is ready.'#, result
		return result

	def getVal(self):
		self.val_list = []
		f = open(self.test_path, "r")
		for line in f:
			words = line.split()
			pos = f.next().split()
			tags = f.next().split()
			self.val_list = self.val_list + tags

		print 'val list is ready.'#, self.val_list

	def validate(self, l1, l2, l3):
		result = self.train(l1, l2, l3)
		self.getVal()
		val_result = self.process(self.val_list)
		num_correct = 0
		num_pre = 0
		num_val = 0
		for key in val_result.keys():
			pre = result[key]
			val = val_result[key]
			num_pre += len(pre)
			num_val += len(val)
			val_set = Set(val)
			for idx in pre:
				if idx in val_set:
					num_correct += 1
		precision = (num_correct * 1.0) / num_pre
		recall = (num_correct * 1.0) / num_val
		F_1 = 2.0 * precision * recall / (precision + recall)
		print 'precision: ', precision, ', recall: ', recall, ', F-1 score: ', F_1
		return F_1


	def viterbi_singleLine(self,line, poses, l1,l2,l3):

	    def possible_tags(k):
	    	if k == 0 or k == -1:
	    		return ['<start>']
	    	else:
	    		return self.tags_list	
	    
	    pi = defaultdict(lambda:0)
	    tag_path = defaultdict()
	    pi[(0,'<start>','<start>')] = 0
	    tag_path[('<start>','<start>')] = list() 

	    n = len(line)

	    ## handle case when only 1 word
	    if n == 1:
	    	word = line[0]
	    	if word in self.word_dict:
	    		word = word
	    	elif poses[0] in self.word_dict:
	    		word = poses[0]
	    	else:
	    		word = '<UNK>'
	    	max_p = float("-inf")
	    	max_t = ''
	    	for t in self.tags_list:
	    		p = 1.0 * self.word_tag_cnt[(t,word)] / self.word_dict[word] 
	    		if p > max_p:
	    			max_p = p
	    			max_t = t
	    	return [t]

	    for k in range(1, n + 1):
	    	temp_bp = {}
	    	word = line[k-1]
	    	if word in self.word_dict:
	    		word = word
	    	elif poses[k - 1] in self.word_dict:
	    		word = poses[ k - 1]
	    	else:
	    		word = '<UNK>'

	    	for u in possible_tags(k-1):
	    		for v in possible_tags(k):
	    			temp_pi = []
	    			for w in possible_tags(k-2):
	    				interpolated = l3*self.transition_tri[(w,u,v)] + l2 * self.transition_bi[(u,v)] + l1 * self.transition_uni[v]
	    				temp_pi.append(pi[k-1,w,u] + np.log(self.emission_prob[(v,word)]) + np.log(interpolated)) 	
	    			pi[(k,u,v)] = max(temp_pi)
	    			max_idx = temp_pi.index(max(temp_pi))
	    			max_w = possible_tags(k-2)[max_idx]
	    			temp_bp[u,v] = tag_path[max_w,u] + [v]
	    	tag_path = temp_bp


	    max_u = ''
	    max_v = ''
	    max_pi = float("-inf")
	    for u in self.tags_list:
	    	for v in self.tags_list:
	    		if pi[n,u,v] > max_pi:
	    			max_pi = pi[n,u,v]
	    			max_u = u
	    			max_v = v
	    return tag_path[max_u,max_v]

	def interpolate(self, lambda1_list, lambda2_size):

	    acc = np.zeros((len(lambda1_list), lambda2_size))
	    
	    self.getNgram()
	    self.recalulate_bigram = False # avoid recalculate bigram
	    
	    for i in range(len(lambda1_list)):
	    	l3 = lambda1_list[i]
	    	l2_range =  1.0 - l3
	    	delta = l2_range * 1.0 / (lambda2_size - 1)
	        for j in range(lambda2_size):
	        	l2 = j * delta
	        	l1 = l2_range -  l2
	        	acc[i, j] = self.validate(l1, l2, l3)
	    idx1 = np.argmax(acc)
	    jj = idx1 % lambda2_size
	    ii = idx1 / lambda2_size
	    l3 = lambda1_list[ii]
	    l2 = jj * (1.0 - l3) / (lambda2_size - 1)
	    l1 = 1.0 - l3 - l2
	    print 'best acc: ', acc[ii, jj]
	    print 'best lambdas: ', l1, l2, l3 
	    return l1, l2, l3



if __name__ == '__main__':
	'''
	# for validation
	tick = time.time()
	train_path = '../data/train_1.txt'
	test_path = '../data/val.txt'
	val_hmm = HMM(train_path, test_path)
	val_hmm.getNgram()
	val_hmm.validate(0, 0, 1.0)
	toc = time.time()
	print 'this process costs ...', toc - tick, ' s.'
	'''
	
	#for output
	tick = time.time()
	train_path = '../data/train.txt'
	test_path = '../data/test.txt'
	kaggle_hmm1 = HMM(train_path, test_path)
	result = kaggle_hmm1.train(0, 0.12, 0.88)
	out_filename = '../result/hmm_trigram_unk_0.12_0.88.csv'
	kaggle_hmm1.out2csv(result,out_filename)
	toc = time.time()
	print 'this process costs ...', toc - tick, ' time'
	
	'''
	# for interpolation
	tick = time.time()
	train_path = '../data/train_1.txt'
	val_path = '../data/val.txt'
	hmm1 = HMM(train_path, val_path)
	lambda2_size = 4
	lambda3_list = [0.91, 0.88, 0.85, 0.81]

	l1, l2, l3 = hmm1.interpolate(lambda3_list, lambda2_size)
	toc = time.time()
	print 'this process costs ...', toc - tick, ' time'
	'''
	