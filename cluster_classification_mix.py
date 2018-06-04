from clust import RebuildError
from classifier import classifier

class cluster_classification_mix (classifier):
	def __init__ (self, dataset_ref, trlabel, slice=False,
	              Clust_method=None, #Clust_method is a clustering method instance
	              Classify_class=None #Classify_class is a classification method class
	              ):
		super().__init__(dataset_ref, trlabel, slice)
		self.clust = Clust_method
		self.classify_class = Classify_class

	def build(self):
		super().build()
		try:
			self.clust.build()
		except RebuildError:
			pass
		for one_cluster in self.clust.clusters:
			clust_slice = self.clust.clusters[one_cluster]

	def predict (self):
		pass


