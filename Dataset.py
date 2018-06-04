import numpy as np


class Dataset:
	def __init__ (self, matrix=[]):
		self.M = np.matrix(matrix)
		self.n = len(matrix)
		self.mean = None
		self.var = None

	def __len__ (self):
		return self.n

	def __getitem__ (self, item):
		return self.M[item, :]

	def __add__ (self, other):
		return Dataset(np.concatenate((self.M, other.M), axis=0))

	def getT (self):
		return Dataset(self.M.getT())

	def get_mean_array (self):
		if self.mean is None:
			self.mean = np.mean(self.M, axis=0)
		return self.mean

	def get_mean (self):
		return np.mean(self.get_mean_array())

	def get_var_array (self):
		if self.var is None:
			self.var = np.var(self.M, axis=0)
		return self.var

	def get_var (self):
		return sum(self.get_var_array())
