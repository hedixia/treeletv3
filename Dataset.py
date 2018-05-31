import numpy as np 

class Dataset:
	def __init__ (self, matrix):
		self.M = np.matrix(matrix)
		self.n = len(matrix)
		
	def __len__ (self):
		return self.n 
	
	def __getitem__ (self, i):
		return self.M[i,:]
		
	def getT (self):
		return Dataset(self.M.getT())