from classifier import classifier
from sklearn import svm
import numpy as np

class MajorityVote(classifier):
	def build (self):
		super().build()
		self.clf = svm.SVC(decision_function_shape='ovo')
		temp_dat = [self.dataset_ref[i] for i in self.slice]
		temp_lab = [self.trlabel[i] for i in self.slice]
		self.clf.fit(temp_dat, temp_lab)
		temp_trn = self.clf.predict(temp_dat)
		self.trerr = np.mean(np.array(temp_lab) == temp_trn)

	def predict (self, test_data):
		return self.clf.predict([test_data])[0]

