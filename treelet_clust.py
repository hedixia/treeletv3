import numpy as np

from clust import clust
from treelet import treelet


class treelet_clust(clust):
	def __init__ (self, dataset_ref, kernel, slice=False, num_clust=1, all_kernel=False):
		super().__init__(dataset_ref, slice)
		self.kernel = kernel
		if num_clust < 1:
			raise ValueError
		self.num_clust = num_clust
		if all_kernel:
			temp_slice = np.array(self.slice, dtype=np.intp)
			self.A = all_kernel[temp_slice[:, np.newaxis], temp_slice]
		else:
			temp_f = lambda i,j : self.kernel(self.dataset_ref[i], self.dataset_ref[j])
			self.A = np.vectorize(temp_f)(*np.meshgrid(self.slice, self.slice, sparse=True))

	def build (self):
		if self.size is 0:
			raise ValueError
		trl = treelet(self.A, self.psi)
		trl.fullrotate()
		self.cltree = trl.tree()
		temp_labels = list(range(self.size))
		for i in range(self.size - self.num_clust):
			temp_labels[self.cltree[i][1]] = self.cltree[i][0]
		for i in range(self.size):
			current = i
			while current != temp_labels[current]:
				current = temp_labels[current]
			ending = current
			current = i
			while current != ending:
				temp_labels[current] = ending
				current = temp_labels[current]
		self.labels = dict(zip(self.slice, temp_labels))
		for i in range(len(trl.dendrogram_list)):
			print(i, trl.dendrogram_list[i])
		self.temp_labels = temp_labels
		self._l2c()

	def assign (self, data):
		linkf = lambda x:self.kernel(self.dataset_ref[x], data)
		closest = max(self.slice, key=linkf)
		return self.labels[closest]


"""
tc = treelet_clust(dat, ker, slice, num_clust)
tc.build()
tc.get()
"""