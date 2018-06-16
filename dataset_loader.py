from Dataset import Dataset

import csv
import numpy as np

class loadDataset (Dataset):
	def __init__ (self, datadir, filenames, **kwargs):
		"""
		:param datadir: directory of datasets
		:param filenames: filenames of datasets
		:param kwargs: keywords
			datatype: datatype of the imported data
		"""
		super().__init__(self, **kwargs)
		self.datadir = datadir
		self.filenames = [filenames] if type(filenames) is str else filenames
		for filename in self.filenames:
			self.load(filename)

	def load (self, filename):
		try:
			datatype = self.datatype
		except AttributeError:
			datatype = float
		csvfile = open(self.datadir + filename)
		newdata = np.matrix(csv.reader(csvfile))
		newdata.astype(datatype, copy=True)
		if len(y) is 0:
			self.M = newdata
		else:
			self.M = np.vstack([self.M, newdata])
		self.n += len(newdata)