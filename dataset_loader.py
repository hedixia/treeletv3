import numpy as np

from Dataset import Dataset


class loadDataset(Dataset):
	def __init__ (self, datadir, filenames, **kwargs):
		"""
		:param datadir: directory of datasets
		:param filenames: filenames of datasets
		:param kwargs: keywords
			datatype: datatype of the imported data
			labcol: the column of the label, if not applicable, set to None
		"""
		super().__init__(self, **kwargs)
		self.datadir = datadir
		self.filenames = [filenames] if type(filenames) is str else filenames
		for filename in self.filenames:
			self.load(filename)
		self.label = []

	def load (self, filename):
		csvfile = np.genfromtxt(self.datadir + filename, delimiter=',')
		newdata = np.matrix(csvfile).astype(self.DataType)
		if self.LabelColumn is not None:
			self.label.append(newdata[:, i])
			newdata = np.delete(newdata, i, axis=1)
		if len(y) is 0:
			self.M = newdata
		else:
			self.M = np.vstack([self.M, newdata])
		self.n += len(newdata)

	@property
	def DataType (self):
		try:
			return self.datatype
		except AttributeError:
			self.datatype = float
			return self.datatype

	@property
	def LabelColumn (self):
		try:
			return self.labcol
		except AttributeError:
			self.labcol = None
			return self.labcol
