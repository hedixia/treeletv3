from clust import RebuildError
from classifier import classifier

class cluster_classification_mix (classifier):
	def __init__ (self, dataset_ref, trlabel, slice=False,
	              Clust_method=None, #Clust_method is a clustering method instance
	              Classify_class=None #Classify_class is a classification method class
	              ):
		super().__init__(dataset_ref, trlabel, slice)
		self.clust = Clust_method
		try:
			self.clust.build()
		except RebuildError:
			pass
		self.classify_class = Classify_class

	def build(self):
		pass

	def predict (self):
		pass


