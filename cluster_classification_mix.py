from clust import RebuildError
from classifier import classifier

class cluster_classification_mix (classifier):
	def __init__ (self, dataset_ref, trlabel, slice=False,
	              Clust_method=None, #a clustering method instance
	              Classify_class=None, #a classification method class
	              Classify_class_kwargs=[] #arguments for classifier class
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
		self.clusterwise_CLM= {}
		for one_cluster in self.clust.clusters:
			clust_slice = self.clust.clusters[one_cluster]
			self.clusterwise_CLM[one_cluster] = classifier(self.dataset_ref, self.trlabel, clust_slice)
			self.clusterwise_CLM[one_cluster].down_cast(self.classify_class, self.classify_class)
			self.clusterwise_CLM[one_cluster].build()
		self.trerr = sum([self.clusterwise_CLM[i].size * self.clusterwise_CLM[i].training_error() for i in self.clusterwise_CLM]) / self.size

	def predict (self, test_data):
		clust = self.clust.assign(test_data)
		return self.clusterwise_CLM[clust].predict(test_data)

