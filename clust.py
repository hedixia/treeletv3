import numpy as np


class clust:
	def __init__ (self, dataset_ref=[], slice=False):
		self.dataset_ref = dataset_ref
		self.input_slice(slice)

	def build (self):
		self.labels = {}
		self.clusters = {}

	def get (self, return_type="clusters"):
		if return_type in ["C", "clusters"]:
			return self.clusters
		if return_type in ["L", "labels"]:
			return self.labels
		return None

	def show (self, title=False):
		if title:
			print("\n" + title + "\n")
		else:
			print("\nClusters\n")
		for i in self.clusters:
			print(i, self.clusters[i])

	def input_slice (self, slice, out=False):
		if slice is False:
			size = len(self.dataset_ref)
			slice = range(size)
		elif type(slice) is int:
			size = slice
			slice = range(slice)
		else:
			size = len(slice)
			slice = slice
		if out:
			return size, slice
		else:
			self.size = size
			self.slice = slice

	def __len__ (self):
		return self.size

	def _l2c (self):
		self.clusters = {}
		for i in self.labels:
			self.clusters.setdefault(self.labels[i], []).append(i)

	def _c2l (self):
		self.labels = {}
		for i in self.clusters:
			for j in self.clusters[i]:
				self.labels[j] = i

	def psi (self, x, y, z):
		return np.abs(x) / np.sqrt(np.abs(y * z))
