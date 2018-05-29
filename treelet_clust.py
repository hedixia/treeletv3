import numpy as np 
from treelet import treelet
import qfs

class treelet_clust:
	def __init__ (self, dataset, slice=False, kernel=False):
		self.K = kernel
		self.dataset = np.array(dataset)
		self.n = self.dataset.shape[0]
		self.__slice = slice if slice else range(self.n)
		self.psi = lambda x,y,z : np.abs(x) / np.sqrt(np.abs(y*z))

	def cor(self, A=False):
		if A:
			self.A = A
		#This step can be speeded up with parallel computing
		self.A = np.array([[self.K(self.dataset[i,:], self.dataset[j,:]) for i in self.__slice] for j in self.__slice])
		
	def tree(self, cor=False):
		if not hasattr(self, 'A'):
			self.cor()
		trl = treelet(self.A, self.psi)
		trl.fullrotate()
		self.cltree = trl.tree()
		
	def clusters(self, num_clust, return_type="L"):
		#requires self.tree being called
		n = self.n
		labels = [i for i in range(n)]
		for i in range(n - num_clust):
			labels[self.cltree[i][1]] = self.cltree[i][0]
		for i in range(n):
			current = i
			while True:
				if current == labels[current]:
					break 
				else:
					current = labels[current]
			ending = current
			current = i
			while current != ending:
				next = labels[current]
				labels[current] = ending
				current = next
		if return_type == "L":
			return labels
		else:
			return qfs.l2dl(labels)
		
"""
tc = treelet_clust(dat)
tc.K = Kernel
tc.cor() #this step may be ignored
tc.tree()
tc.clusters(num)
"""