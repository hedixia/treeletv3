from clust import clust


class classifier(clust):
	def __init__ (self, dataset_ref=[], trlabel=[], slice=False):
		super().__init__(dataset_ref, slice)
		self.trlabel = trlabel
		self.true_ratio = None

	def build (self):
		super().build()
		if len(self) is 0:
			raise ValueError("Empty dataset.")
		if len(self.dataset_ref) != len(self.trlabel):
			errorstring = "Length of dataset ({0}) is not equal to the length of its label ({1}).".format(len(self.dataset_ref), len(self.trlabel))
			raise ValueError(errorstring)

	def predict (self, test_data):
		return self.trlabel[0]

	def predict_multiple (self, test_dataset, slice=False):
		temp = clust(test_dataset, slice)
		return {x:self.predict(test_dataset[x]) for x in temp.slice}

	def training_error (self):
		return 1 - self.true_ratio

	def down_cast (self, subclass_name, args, **kwargs):
		self.__class__ = subclass_name
		for key in args:
			setattr(self, key, args[key])
		for key in kwargs:
			setattr(self, key, kwargs[key])