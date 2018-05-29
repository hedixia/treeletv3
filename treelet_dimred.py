import numpy as np 
from treelet import treelet

class treelet_dimred:
	def __init__ (self, dataset):
		self.dataset = np.matrix(dataset)
		self.datasetT = dataset.getT()
	
	def cov(self, A=False):
		if A:
			self.cov = A 
		else:
			self.cov = np.cov(self.datasetT)
		
	def tree(self, t=0):
		psi = lambda x,y,z : abs(x)/np.sqrt(np.abs(y * z)) + abs(x) * t
		self.trl = treelet(self.cov, psi)
		self.n = self.trl.n 
		self.transform_list = self.trl.transform_list
		self.dfrk = self.trl.dfrk
		
	#Treelet Transform
	def forw (self, v, k=None, epsilon=None):
		if k == None:
			k = self.n - 1
		for iter in range(k):
			(scv, cgs, cos_val, sin_val) = self.transform_list [iter]
			temp_scv = cos_val * v[scv] - sin_val * v[cgs]
			temp_cgs = sin_val * v[scv] + cos_val * v[cgs]
			v[scv] = temp_scv
			v[cgs] = temp_cgs
		if epsilon == None:
			return v
		else:
			newdict = {}
			newvec = np.zeros(n-k)
			for i in range(n):
				if i < k:
					if abs(v[self.dfrk[i]]) > epsilon:
						newdict[self.dfrk[i]] = v[self.dfrk[i]]
				else:
					newvec[i] = v[self.dfrk[i+k]]
			return (newvec, newdict)
			
	def back (self, v, diffdict={}):
		k = self.n - len(v)
		if k != 0:
			newv = np.zeros(self.n)
			for i in range(len(v)):
				newv[self.dfrk[i+k]] = v[i]
			v = newv 
		for iter in reversed(self.transform_list):
			(scv, cgs, cos_val, sin_val) = iter
			temp_scv = cos_val * v[scv] + sin_val * v[cgs]
			temp_cgs = -sin_val * v[scv] + cos_val * v[cgs]
			v[scv] = temp_scv
			v[cgs] = temp_cgs
		for i in diffdict:
			v[i] += diffdict[i]
		return v