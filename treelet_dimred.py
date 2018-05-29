import numpy as np 
from treelet import treelet

class treelet_dimred:
	def __init__ (self, dataset_ref, cov=False):
		self.dataset_ref = dataset_ref
		self.cov = cov if cov else np.cov(np.matrix(dataset_ref).getT())
		
	def build (self, t=0):
		psi = lambda x,y,z : abs(x)/np.sqrt(np.abs(y * z)) + abs(x) * t
		self.trl = treelet(self.cov, psi)
		self.n = self.trl.n 
		self.transform_list = self.trl.transform_list
		self.dfrk = self.trl.dfrk
		
	#Treelet Transform
	def forw (self, v, k=False, epsilon=0):
		k = k if k else self.n - 1
		for iter in range(k):
			(scv, cgs, cos_val, sin_val) = self.transform_list [iter]
			temp_scv = cos_val * v[scv] - sin_val * v[cgs]
			temp_cgs = sin_val * v[scv] + cos_val * v[cgs]
			v[scv] = temp_scv
			v[cgs] = temp_cgs
		if epsilon == 0:
			return [v, None]
		else:
			newdict = {}
			newvec = np.zeros(n-k)
			for i in range(n):
				if i < k:
					if abs(v[self.dfrk[i]]) > epsilon:
						newdict[self.dfrk[i]] = v[self.dfrk[i]]
				else:
					newvec[i] = v[self.dfrk[i+k]]
			return [newvec, newdict]
			
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