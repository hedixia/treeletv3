from treelet_clust import treelet_clust
from random import sample
import qfs
import numpy


class SA_clust(treelet_clust):
	def __init__ (self, dataset_ref, kernel, sample_para, slice=False, num_clust=0, all_kernel=False):
		self.K = kernel
		self.all = dataset_ref
		self.__enter_slice(slice)
		if type(sample_para) == int:
			if len(self.__slice) < sample_para:
				sample_para = len(self.__slice)
			self.sample_index = sample(self.__slice, sample_para)
		else:
			self.sample_index = sample_para
		sample_data = [dataset_ref[i] for i in self.sample_index]
		super().__init__(dataset_ref, kernel, slice, num_clust, all_kernel)
		
	def build (self, return_type="L"):
		super().build()
		self.labels = dict(zip(self.__slice, self.__assignment(self.clusters)))
		self.clusters = qfs.l2c(self.labels)

	def __enter_slice (self, slice):
		if type(slice) == int:
			self.size = slice
			self.__slice = [i for i in range(slice)]
		else:
			self.size = len(slice)
			self.__slice = slice

	def __aff (self, v, w, index=(True, True), clusters=False):
		if clusters:
			return max([self.__aff(x, w, index) for x in v])
		else:
			return self.psi(self.__inner(v, w, index), self.__inner(v, v, index), self.__inner(w, w, index))

	def __inner (self, v, w, index=(True, True)):
		if index[0]:
			v = self.all[v]
		if index[1]:
			w = self.all[w]
		return self.K(v, w)

	def __assignment (self, clust_dict):
		return_list = [None] * self.size
		clust_name = list(clust_dict)
		temp_score = {}
		for one_data_index in range(len(self.__slice)):
			one_data = self.__slice[one_data_index]
			for one_cluster in clust_name:
				temp_score[one_cluster] = self.__aff(clust_dict[one_cluster], one_data, clusters=True)
			assign_clust = max(temp_score, key=temp_score.get)
			return_list[one_data_index] = assign_clust
		return return_list

	
