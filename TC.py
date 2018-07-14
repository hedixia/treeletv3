import treelet_clust
from Dataset import Dataset
import kernel
import numpy as np

class TC:
	def __init__ (self, ker, num):
		if ker == "poly":
			self.ker =  kernel.kernel("poly", [1, 3])
		elif ker == "rbk":
			self.ker = kernel.kernel("rbk", [0.1])
		self.num = num
		
	def fit (self, X):
		dX = Dataset(X)
		dker = self.ker
		xTC = treelet_clust.treelet_clust(dX, dker, num_clust=self.num)
		xTC.build()
		
		tempL = sorted(list(set(xTC.labels_)))
		tempD = {tempL[x]:x for x in range(len(tempL))}
		self.labels_ = np.array([tempD[i] for i in xTC.labels_])