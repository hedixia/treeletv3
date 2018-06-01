class clust:
	def __init__ (self):
		pass

	def build (self):
		self.labels = {}
		self.clusters = {}

	def get (self, return_type="clusters"):
		if return_type in ["C", "clusters"]:
			return self.clusters
		if return_type in ["L", "labels"]:
			return self.labels
		return None

	def show (self, title=False):
		if title:
			print ("\n" + title + "\n")
		else:
			print("\nClusters\n")
		for i in self.clusters:
			print(i, self.clusters[i])

	def _l2c (self):
		self.clusters = {}
		for i in self.labels:
			self.clusters.setdefault(self.labels[i], []).append(i)

	def _c2l (self):
		self.labels = {}
		for i in self.clusters:
			for j in self.clusters[i]:
				self.labels[j] = i
