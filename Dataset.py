import numpy as np 

class Dataset:
	def __init__ (self, matrix=[]):
		self.M = np.matrix(matrix)
		self.n = len(matrix)

	def __len__ (self):
		return self.n

	def __getitem__ (self, item):
		return self.M[item, :]

	def __add__ (self, other):
		return Dataset(np.concatenate((self.M, other.M), axis=0))

	def getT (self):
		return Dataset(self.M.getT())

