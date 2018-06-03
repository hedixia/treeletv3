import numpy as np
from collections import Counter
from treelet import treelet
from treelet_clust import treelet_clust
from classifier import classifier


class MajorityVote(classifier):
	def build (self):
		super().build()
		cnt = Counter()
		for i in self.slice:
			cnt[self.trlabel[i]] += 1
		temp = cnt.most_common(1)[0]
		self.labels = temp[0]
		self.trerr = temp[1] / self.size

	def predict (self, test_data):
		return self.labels

	def predict_multiple (self, test_dataset, slice=False):
		temp = classifier(slice=slice)
		return {x:self.labels for x in temp.slice}


class treelet_classifier(classifier):
	def __init__ (self, dataset_ref, kernel, trlabel, slice=False, CLM=MajorityVote, all_kernel=False, majority_edge=1):
		super().__init__(dataset_ref, trlabel, slice)
		self.clust = treelet_clust(dataset_ref, kernel, slice, 0, all_kernel)
		self.trlabel = trlabel
		self.CLM = CLM  # prediction = CLM(training_set, training_label, slice).predict(test_data)
		if self.purity_cut(majority_edge):
			raise ValueError

	def build (self):
		super().build()
		trl = treelet(self.clust.A, self.clust.psi)
		trl.fullrotate()
		self.cltree = trl.tree()
		reject_list = [False] * self.size
		clusters = {x:[x] for x in range(self.size)}
		errorlist = [0] * self.size
		self.clusterwise_CLM = {x : MajorityVote(self.dataset_ref, self.trlabel, slice=[slice[x]]) for x in range(self.size)}
		for tup in self.cltree:
			if reject_list[tup[0]]:
				continue
			if reject_list[tup[1]]:
				reject_list[tup[0]] = True
				continue
			len_0 = len(clusters[tup[0]])
			len_1 = len(clusters[tup[1]])
			comb_data = clusters[tup[0]] + clusters[tup[1]]
			trCLM = self.CLM(self.dataset_ref, self.trlabel, slice=comb_data)
			trCLM.build()
			new_error_rate = trCLM.training_error() - 1 / (len_0 + len_1)
			if errorlist[tup[1]] + errorlist[tup[0]] <= new_error_rate:
				clusters[tup[0]] = comb_data
				self.clusterwise_CLM[tup[0]] = trCLM
				del clusters[tup[1]]
				del self.clusterwise_CLM[tup[1]]
				errorlist[tup[0]] = new_error_rate
			else:
				reject_list[tup[0]] = True
		self.clusters = {[slice[j] for j in clusters[i]] for i in clusters}
		self._c2l()

	def predict (self, test_data):
		super().predict(test_data)
		cl = self.assign(test_data)
		return self.clusterwise_CLM[cl].predict(test_data)

	def assign (self, data):
		linkf = lambda x:self.kernel(self.dataset_ref[x], data)
		closest = max(self.slice, key=linkf)
		return self.labels[closest]

	def purity_cut (self, majority_edge):
		if majority_edge == 1:
			return False
		cnt = Counter()
		for i in self.slice:
			cnt[self.trlabel[i]] += 1
		return max(cnt.values()) / sum(cnt.values()) > majority_edge
