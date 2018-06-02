import numpy as np 
from collections import Counter
from treelet import treelet
from treelet_clust import treelet_clust

class treelet_classifier (treelet_clust):
	def __init__ (self, dataset_ref, kernel, trlabel, slice=False, CLM=MajorityVote, all_kernel=False):
		super().__init__(dataset_ref, kernel, slice, 0, all_kernel)
		self.trlabel = trlabel
		self.CLM = CLM
		#prediction = CLM(training_set, training_label, slice=range(len(training_set)))(test_data)
		
	def build (self):
		trl = treelet(self.clust.A, self.clust.psi)
		trl.fullrotate()
		self.cltree = trl.tree()
		rjlist = [False] * self.size
		clustlist = np.arange(self.size, dtype=np.intp)
		weightlist = [0] * self.size
		for tup in self.cltree:
			if rjlist[tup[0]]:
				continue 
			if rjlist[tup[1]]:
				rjlist[tup[0]] = True
				continue 
			len_0 = np.sum(clustlist == tup[0])
			len_1 = np.sum(clustlist == tup[1])
			newdata = [self.dataset_ref[i] for i in range(self.size) if clustlist[i] in tup]
			newlab = [self.trlabel[i] for i in range(self.size) if clustlist[i] in tup]
			trpred = self.CLM(newdata, newlab)(newdata)
			trerr = np.mean(np.array(trpred) == np.array(newlab))
			newweight = trerr - 1/ (len_0 + len_1)
			if weightlist[tup[1]] + weightlist[tup[0]] <= newweight:
				clustlist[tup[1]] = tup[0]
				weightlist[tup[0]] = newweight
			else:
				rjlist[tup[0]] = True
		self.labels = dict(zip(self.slice, clustlist))
		self._l2c()
	
	def predict (self, test_dataset, clust_info=False):
		test_label = [None for i in test_dataset]
		cluster_assignment = [self.assign(each_data) for each_data in test_dataset]
		cluster_tsdata = dict.fromkeys(set(cluster_assignment), [])
		for i in range(len(cluster_assignment)):
			cluster_tsdata[cluster_assignment[i]].append(i)
		for one_cluster in cluster_tsdata:
			training_set = [self.dataset[i] for i in self.cluster_trdata[one_cluster]]
			training_label = [self.trlabel[i] for i in self.cluster_trdata[one_cluster]]
			test_data = [self.dataset[i] for i in cluster_tsdata[one_cluster]]
			somelabels = self.CLM(training_set, training_label)(test_data)
			for i in range(len(cluster_tsdata[one_cluster])):
				test_label[cluster_tsdata[one_cluster][i]] = somelabels[i]
		if clust_info:
			return cluster_assignment, cluster_tsdata
		return test_label
		
	def assign (self, data):
		linkf = lambda x : self.kernel(self.dataset_ref[x], data)
		closest = max(self.slice, key=linkf)
		return self.labels[closest]
		
	def purity (self, slice=None):
		cnt = Counter()
		if slice == None:
			slice = self.slice
		for i in slice:
			cnt[self.trlabel[i]] += 1
		return max(cnt.values())/sum(cnt.values())