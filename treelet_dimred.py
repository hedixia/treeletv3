import numpy as np

from treelet import treelet


def listgen (L, start):
	i = start
	while L[i] != i:
		yield i
		i = L[i]


class treelet_dimred:
	def __init__ (self, t=0):
		self.t = t
		self.n = 0

	def fit (self, dataset_ref):
		self.dataset_ref = np.matrix(dataset_ref)
		self.avedat = np.average(self.dataset_ref, axis=0)
		self.cov = np.cov(self.dataset_ref.getT())
		psi = lambda x, y, z:abs(x) / np.sqrt(np.abs(y * z)) + abs(x) * self.t
		self.trl = treelet(self.cov, psi)
		self.n = self.trl.n
		self.transform_list = self.trl.transform_list
		self.dfrk = self.trl.dfrk

	# Treelet Transform
	def transform (self, v, k=False, epsilon=0):
		v = np.array(v) - self.avedat
		k = k if k else self.n - 1
		for iter in range(k):
			(scv, cgs, cos_val, sin_val) = self.transform_list[iter]
			temp_scv = cos_val * v[scv] - sin_val * v[cgs]
			temp_cgs = sin_val * v[scv] + cos_val * v[cgs]
			v[scv] = temp_scv
			v[cgs] = temp_cgs
		if epsilon == 0:
			return [v, None]
		else:
			newdict = {}
			newvec = np.zeros(n - k)
			for i in range(n):
				if i < k:
					if abs(v[self.dfrk[i]]) > epsilon:
						newdict[self.dfrk[i]] = v[self.dfrk[i]]
				else:
					newvec[i] = v[self.dfrk[i + k]]
			return (newvec, newdict)

	def inverse_transform (self, v, diffdict={}):
		k = self.n - len(v)
		if k != 0:
			newv = np.zeros(self.n)
			for i in range(len(v)):
				newv[self.dfrk[i + k]] = v[i]
			v = newv
		for iter in reversed(self.transform_list):
			(scv, cgs, cos_val, sin_val) = iter
			temp_scv = cos_val * v[scv] + sin_val * v[cgs]
			temp_cgs = -sin_val * v[scv] + cos_val * v[cgs]
			v[scv] = temp_scv
			v[cgs] = temp_cgs
		for i in diffdict:
			v[i] += diffdict[i]
		return v + self.avedat

	def cluster (self, k):
		returnL = [i for i in range(self.n)]
		for i in range(k):
			returnL[self.transform_list[i][1]] = self.transform_list[i][0]
		for i in range(n):
			if returnL[i] == i:
				continue
			if returnL[returnL[i]] == returnL[i]:
				continue
			tempL = list(listgen(returnL, i))
			for j in tempL:
				returnL[j] = tempL[-1]
		return returnL

	def __len__ (self):
		return self.n

	self.__call__ = self.transform
