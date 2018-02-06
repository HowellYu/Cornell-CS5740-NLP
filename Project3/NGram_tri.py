dimport numpy as np
from collections import defaultdict

class ngram():
	def __init__(self, train_path):
	    self.train_path = train_path
	    #self.test_path = test_path

	    self.tags_dict = {'O':0,'B-PER' : 1,'I-PER':2,'B-LOC': 3,'I-LOC':4,'B-ORG' : 5,'I-ORG':6,'B-MISC':7,'I-MISC':8}
	    self.word_dict = None

	    self.tags_cnt = None
	    self.tags_bi_cnt = None

	    self.tags_tri_cnt = None
	    
	    self.word_cnt = None
	    self.word_tag_cnt = None

	    self.start_tag_cnt = None
	    self.init_prob = None

	    self.emission_prob = None
	    self.transition_bi = None
	    self.transition_tri = None
	    self.transition_uni = None
	    self.N = 0 # number of lines 


	def get_transition_uni(self):
		self.transition_uni = {}
		total = sum(self.tags_cnt.values())
		for tag,cnt in self.tags_cnt.items():
			self.transition_uni[tag] = 1.0 * cnt / total

	
	# return tags cnts of every line
	#        bigram of labels
	#        unigram of tokens
	def first_pass(self):
	    self.start_tag_cnt = defaultdict(lambda:0)
	    self.word_cnt = defaultdict(lambda:0)
	    self.tags_bi_cnt = defaultdict(lambda:0)
	    self.tags_cnt = defaultdict(lambda:0)
	    self.tags_tri_cnt = defaultdict(lambda:0)
	    
	    f = open(self.train_path, "r")
	    for line in f:
	        words = line.split()
	        poses = f.next().split()
	        tags = f.next().split()
	        
	        self.N += 1
	        t1 = '<start>'
	        t2 = '<start>'
	        t3 = '<start>'

	        for i in range(len(words)):
	        	t1 = t2
	        	t2 = t3
	        	t3 = tags[i]
	        	if t3 in self.tags_cnt:
	        		self.tags_cnt[t3] += 1
	        	else:
	        		self.tags_cnt[t3] = 1
	        	
	        	bi = (t2 ,t3) 
	        	if bi in self.tags_bi_cnt:
	        		self.tags_bi_cnt[bi] += 1
	        	else:
	        		self.tags_bi_cnt[bi] = 1

	        	tri = (t1,t2,t3)
	        	if tri in self.tags_tri_cnt:
	        		self.tags_tri_cnt[tri] += 1
	        	else:
	        		self.tags_tri_cnt[tri] = 1	      

	        for word in words:
	            if word not in self.word_cnt:
	                self.word_cnt[word] = 1
	            else:
	                self.word_cnt[word] += 1
	        
	    f.close()

	def sec_pass(self):
	    self.word_tag_cnt = defaultdict(lambda:0)
	    unknown = '<UNK>'
	    f = open(self.train_path, "r")
	    for line in f:
	        words = line.split()
	        poses = f.next().split()
	        tags = f.next().split()
	        
	        n = len(words)
	        
	        for i in range(n):
	            if self.word_cnt[words[i]] != 1:
	                cur = words[i]
	            else:
	                #cur = unknown 
	                cur = poses[i]	# change unknown to its pos tag
	            	self.word_cnt.pop(words[i])
	            	if cur in self.word_cnt:
	            		self.word_cnt[cur] += 1
	            	else:
	            		self.word_cnt[cur] = 1

	            bi = (tags[i], cur)
	            if bi in self.word_tag_cnt:
	                self.word_tag_cnt[bi] += 1
	            else:
	                self.word_tag_cnt[bi] = 1
	    f.close()
	    return self.word_tag_cnt

	def get_word2idx(self, k1, k2):
		self.word_dict = defaultdict(lambda:0)
		cnt = 0
		sth = self.word_cnt.copy()
		for v in sth.keys():
		    self.word_dict[v] = cnt
		    cnt += 1

	def get_emission(self):
		self.emission_prob = defaultdict(lambda:np.random.uniform(10**(-10), 10**(-11)))
		for tagWord, count in self.word_tag_cnt.items():
			tag = tagWord[0]
			tagCount = self.tags_cnt[tag]
			self.emission_prob[tagWord] = 1.0 * count / tagCount 

	def get_transition(self):
			self.transition_bi = defaultdict(lambda:np.random.uniform(10**(-10), 10**(-11)))
			self.transition_tri = defaultdict(lambda:np.random.uniform(10**(-10), 10**(-11)))
			for bi, count in self.tags_bi_cnt.items():
				t1 = bi[0]
				if t1 == '<start>':
					preCount = self.N
				else:
					preCount = self.tags_cnt[t1]
				self.transition_bi[bi] = 1.0 * count / preCount
			
			for tri, count in self.tags_tri_cnt.items():
				t1t2 = tri[0:2]
				if t1t2 == ('<start>','<start>'):
					preCount = self.N
				else:
					preCount = self.tags_bi_cnt[t1t2]
				self.transition_tri[tri] = 1.0 * count / preCount



if __name__ == "__main__":
	train_path = '../data/train.txt'
	trigram = ngram(train_path)
	trigram.first_pass()
	trigram.sec_pass()		
	trigram.get_transition()
	trigram.get_emission()
	trigram.get_transition_uni();
	#trigram.cnt2prob()
	trigram.get_prob(k1 = 1.0, k2 = 1.0)
	print 'ngram is done.'
