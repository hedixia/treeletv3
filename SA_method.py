from random import sample

from clust import clust
from treelet_clust import treelet_clust


class SA_clust(clust):
	def __init__ (self, dataset_ref, kernel, sample_para, slice=False, num_clust=0, all_kernel=False):
		super().__init__(dataset_ref, slice)
		self.kernel = kernel
		if type(sample_para) is int:
			if self.size < sample_para:
				sample_para = self.size
			sample_index = sample(self.slice, sample_para)
		else:
			sample_index = sample_para
		self.trcl = treelet_clust(dataset_ref, kernel, sample_index, num_clust, all_kernel)
		
	def build (self):
		self.trcl.build()
		self.labels = {}
		clust_name = list(self.trcl.clusters)
		temp_score = {}
		for one_data in self.slice:
			for one_cluster in clust_name:
				temp_score[one_cluster] = self.__aff(self.trcl.clusters[one_cluster], one_data, clusters=True)
			self.labels[one_data] = max(temp_score, key=temp_score.get)
		self._l2c()

	def __aff (self, v, w, clusters=False):
		if clusters:
			return max([self.__aff(x, w) for x in v])
		else:
			return self.psi(self.__inner(v, w), self.__inner(v, v), self.__inner(w, w))

	def __inner (self, v, w):
		v = self.dataset_ref[v]
		w = self.dataset_ref[w]
		return self.kernel(v, w)

