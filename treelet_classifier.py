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
		clusters = {x:[x] for x in range(self.size)}
		weightlist = [0] * self.size
		for tup in self.cltree:
			if rjlist[tup[0]]:
				continue 
			if rjlist[tup[1]]:
				rjlist[tup[0]] = True
				continue 
			len_0 = len(clusters[tup[0]])
			len_1 = len(clusters[tup[1]])
			comb_data = clusters[tup[0]] + clusters[tup[1]]
			trCLM = self.CLM(self.dataset_ref, self.trlabel, slice=comb_data)
			trCLM.build()
			trerr = trCLM.training_error()
			newweight = trerr - 1/ (len_0 + len_1)
			if weightlist[tup[1]] + weightlist[tup[0]] <= newweight:
				clusters[tup[0]] = comb_data
				del clusters[tup[1]]
				weightlist[tup[0]] = newweight
			else:
				rjlist[tup[0]] = True
		self.clusters = {[slice[j] for j in clusters[i]] for i in clusters}
		self._c2l()
	
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