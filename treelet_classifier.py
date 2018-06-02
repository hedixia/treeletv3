import numpy as np 
from collections import Counter
from treelet import treelet


class treelet_classifier:
	def __init__ (self, dataset_ref, kernel, label, slice=False, CLM=MajorityVote, all_kernel=False):
		self.clust = treelet_clust(dataset_ref, kernel, slice, 0, all_kernel)
		self.K = kernel
		self.label = label
		self.CLM = CLM
		#prediction = CLM(training_set, training_label, slice=range(len(training_set)))(test_data)
		
	def build (self):
		self.clust.build()
		rjlist = [False for i in range(self.n)]
		clustlist = np.array([i for i in range(self.n)])
		weightlist = [0 for i in range(self.n)]
		for tup in self.cltree:
			if rjlist[tup[0]]:
				continue 
			if rjlist[tup[1]]:
				rjlist[tup[0]] = True
				continue 
			len_0 = np.sum(clustlist == tup[0])
			len_1 = np.sum(clustlist == tup[1])
			newdata = [self.dataset[i] for i in range(self.n) if clustlist[i] in tup]
			newlab = [self.label[i] for i in range(self.n) if clustlist[i] in tup]
			trpred = self.CLM(newdata, newlab)(newdata)
			trerr = compare(trpred, newlab)
			newweight = index(trerr, len_0, len_1)
			if weightlist[tup[1]] + weightlist[tup[0]] <= newweight:
				clustlist[tup[1]] = tup[0]
				weightlist[tup[0]] = newweight
			else:
				rjlist[tup[0]] = True
		self.keys = list(set(clustlist)).sorted()
		self.cluster_trdata = dict.fromkeys(set(clustlist), [])
		for i in range(len(clustlist)):
			self.cluster_trdata[clustlist[i]].append(i)
		self.clustlist = clustlist
		return clustlist
	
	def predict (self, test_dataset, clust_info=False):
		test_label = [None for i in test_dataset]
		cluster_assignment = [self.assign(each_data) for each_data in test_dataset]
		cluster_tsdata = dict.fromkeys(set(cluster_assignment), [])
		for i in range(len(cluster_assignment)):
			cluster_tsdata[cluster_assignment[i]].append(i)
		for one_cluster in cluster_tsdata:
			training_set = [self.dataset[i] for i in self.cluster_trdata[one_cluster]]
			training_label = [self.label[i] for i in self.cluster_trdata[one_cluster]]
			test_data = [self.dataset[i] for i in cluster_tsdata[one_cluster]]
			somelabels = self.CLM(training_set, training_label)(test_data)
			for i in range(len(cluster_tsdata[one_cluster])):
				test_label[cluster_tsdata[one_cluster][i]] = somelabels[i]
		if clust_info:
			return cluster_assignment, cluster_tsdata
		return test_label
		
	def assign (self, data):
		return argmax(self.keys, self.linkf)
		
	def closeness_measurment (self, data):
		kernel_function = lambda x : self.K(self.dataset[x], data)
		if self.link == "inf":
			self.linkf = lambda one_cluster : kernel_function(argmax(self.cluster_trdata[one_cluster], kernel_function))
		else:
			kwf = lambda x : kernel_function(x) ** self.link
			self.linkf = lambda one_cluster : np.mean(lapply(self.cluster_trdata[one_cluster], kwf))
		return self.linkf 
		
	def linkage (self, link):
		self.link = link
		
	def purity (self, slice=None):
		cnt = Counter()
		if slice == None:
			slice = self.slice
		for i in slice:
			cnt[self.label[i]] += 1
		return max(cnt.values())/sum(cnt.values())
	
