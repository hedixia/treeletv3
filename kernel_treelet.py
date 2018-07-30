from treelet_dimred import treelet_dimred

class KernelTreelet:
	def __init__ (self, kernel=False, t=False):
		self.K = self._kernel(kernel)
		self.trl = treelet_dimred(t=t)

	def fit (self, X, k):
		self.A_0 = self.K(X)
		A_0 = self.A_0.copy()
		self.trl.fit(A_0)
		A_k = self.trl.transform(self.trl.transform(self.A_0.getT())[0].getT())[0]
		self.A_k = A_k
		self.L_k = self.decomp(A_k)
		self.Delta_k = self.trl.transform(np.identity(A_0.shape[0]))[0] * self.L_k


	def decomp (self, M):
		return

	def _kernel (self, kernel):
		return