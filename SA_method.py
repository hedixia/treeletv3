from treelet_clust import treelet_clust
from random import sample 
import qfs
import numpy

class SA_clust (treelet_clust):
	def __init__ (self, dataset, slice, sample_para, kernel=None):
		self.K = kernel
		self.all = numpy.matrix(dataset) 
		self.enter_slice(slice)
		if type(sample_para) == int:
			if len(self.__slice) < sample_para:
				sample_para = len(self.__slice)
			self.sample_index = sample(self.__slice, sample_para)
		else:
			self.sample_index = sample_para
		sample_data = [dataset[i] for i in self.sample_index]
		super().__init__(sample_data)
		
	def enter_slice (self, slice):
		if type(slice) == int:
			self.size = slice 
			self.__slice = [i for i in range(slice)]
		else:
			self.size = len(slice)
			self.__slice = slice
		
	def aff (self, v, w, index=(True, True), clusters=False):
		if clusters:
			return max([self.aff(x, w, index) for x in v])
		else:
			return self.psi(self.inner(v, w, index), self.inner(v, v, index), self.inner(w, w, index))
		
	def inner (self, v, w, index=(True, True):
		if index[0]:
			v = self.all[v,:]
		if index[1]:
			w = self.all[w,:]
		return self.K(v,w)
		
	def assignment (self, clust_dict):
		return_list = [None] * self.size
		clust_name = list(clust_dict)
		temp_score = {}
		for one_data_index in range(len(self.__slice)):
			one_data = self.__slice[one_data_index]
			for one_cluster in clust_name:
				temp_score[one_cluster] = self.aff(clust_dict[one_cluster], one_data, True)
			assign_clust = max(temp_score, key=temp_score.get)
			return_list[one_data_index] = assign_clust
		return return_list
		
	def clusters (self, num_clust, return_type="L"):
		self.clust_dict = super().clusters(num_clust, return_type="D")
		clust_list = self.assignment(self.clust_dict)
		if return_type == "L":
			return clust_list
		else:
			return qfs.l2dl(clust_list)