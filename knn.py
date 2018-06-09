from classifier import classifier


class KNN(classifier):
	def __init__ (self, dataset_ref=[], trlabel=[], slice=False,
	              measure=lambda x, y:1, K=0):
		super().__init__(dataset_ref, slice)
		self.K = K

	def build (self):
		super().build()
		if self.K is not 0:
			Tval = 0
			Fval = 0
			for i in self.slice:
				if self.predict(self.dataset_ref[i]) == self.labels[i]:
					Tval += 1
				else:
					Fval += 1
			self.true_ratio = Tval / (Tval + Fval)
		else:
			pass  # Cross Validation to select K

	def predict (self, test_data):
		MTD = {x:self.measure(self.dataset_ref[x], test_data) for x in self.slice}
		first_K = sorted(MTD.keys(), MTD.get)[:self.K]
		K_lab = [self.labels[i] for i in first_K]
		return max(set(K_lab), key=K_lab.count)
