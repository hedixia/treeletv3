import numpy as np


class Dataset:
	def __init__ (self, matrix=[], label=[], **kwargs):
		if matrix.__class__ == self.__class__:
			self.__dict__ = matrix.__dict__
		else:
			self.M = np.matrix(matrix)
			self.n = len(matrix)
			self.mean = None
			self.var = None
			self.label = label

		for key, value in kwargs.items():
			setattr(self, key, value)

	def __len__ (self):
		return self.n

	def __getitem__ (self, item):
		return self.M[item, :]

	@property
	def l (self):
		return self.label

	def __add__ (self, other, otherlabel=[]):
		return Dataset(np.vstack([self.M, other.M]), self.label + otherlabel)

	def getSlice (self, slice):
		if len(self.label) is 0:
			return Dataset(self.M[slice, :])
		else:
			return Dataset(self.M[slice, :], [self.label[i] for i in slice])

	def getT (self):
		return Dataset(self.M.getT())

	def get_mean_array (self):
		if self.mean is None:
			self.mean = np.mean(self.M, axis=0).tolist()[0]
		return self.mean

	@property
	def Mean (self):
		return np.mean(self.get_mean_array())

	def get_var_array (self):
		if self.var is None:
			self.var = np.var(self.M, axis=0).tolist()[0]
		return self.var

	@property
	def Var (self):
		return sum(self.get_var_array())
