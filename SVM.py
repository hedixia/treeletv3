import numpy as np
from sklearn import svm

from classifier import classifier


class SVM(classifier):
	def build (self):
		super().build()
		temp_lab = [self.trlabel[i] for i in self.slice]
		if len(set(temp_lab)) is 1:
			self.val = temp_lab[0]
			self.trerr = 0
			self.degenerate = True
		else:
			temp_dat = self.dataset_ref[self.slice]
			self.clf = svm.SVC(decision_function_shape='ovo')
			self.clf.fit(temp_dat, temp_lab)
			temp_trn = self.clf.predict(temp_dat)
			self.trerr = np.mean(np.array(temp_lab) == temp_trn)
			self.degenerate = False

	def predict (self, test_data):
		return self.val if self.degenerate else self.clf.predict(test_data)[0]
