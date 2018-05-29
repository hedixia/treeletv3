import numpy as np
from treelet import treelet
import qfs


class treelet_clust:
	psi = lambda x, y, z:np.abs(x) / np.sqrt(np.abs(y * z))

	def __init__ (self, dataset_ref, kernel, slice=False, num_clust=0, all_kernel=False):
		self.kernel = kernel
		self.dataset_ref = dataset_ref
		self.__slice = slice if slice else range(len(dataset_ref))
		self.num_clust = num_clust
		if all_kernel:
			temp_slice = np.array(self.__slice, dtype=np.intp)
			self.A = all_kernel[temp_slice[:, np.newaxis], temp_slice]
		else:
			self.A = np.array(
				[[self.kernel(self.dataset[i], self.dataset[j]) for i in self.__slice] for j in self.__slice])

	def build (self):
		trl = treelet(self.A, psi)
		trl.fullrotate()
		self.cltree = trl.tree()
		n = len(self.__slice)
		temp_labels = [i for i in range(n)]
		for i in range(n - self.num_clust):
			temp_labels[self.cltree[i][1]] = self.cltree[i][0]
		for i in range(n):
			current = i
			while current != temp_labels[current]:
				current = temp_labels[current]
			ending = current
			current = i
			while current != ending:
				temp_labels[current] = ending
				current = temp_labels[current]
		self.labels = dict(zip(self.__slice, temp_labels))
		self.clusters = qfs.l2c(self.labels)

	def clusters (self, return_type="C"):
		if return_type == "C":
			return self.clusters
		else:
			return self.labels


"""
tc = treelet_clust(dat, ker, slice, num_clust)
tc.build()
tc.clusters()
"""
