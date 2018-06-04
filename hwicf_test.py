import csv
import os
import random

import numpy as np

from Dataset import Dataset
from kernel import kernel
from treelet_classifier import treelet_classifier

os.chdir(os.path.dirname(os.path.realpath(__file__)))
datadir = r"C:\D\senior_thesis\handwritten_num\samples\comp20data"
trdataextract = {i:100 for i in range(10)}
tsdataextract = {i:1000 for i in range(10)}
	
def labeling (idict):
	L = []
	for i in range(10):
		if i not in idict:
			continue
		L += [i] * idict[i]
	return L

#fetch training data
trL = []
for i in range(10):
	if i not in trdataextract:
		continue
	with open(datadir + r"\train_label_" + str(i) + ".csv") as csvfile:
		reader = csv.reader(csvfile)
		temp = [[int(j) for j in i] for i in reader] 
	trL += random.sample(temp, trdataextract[i])
trL = Dataset(trL)

#fetch test data
tsL = []
for i in range(10):
	if i not in tsdataextract:
		continue
	with open(datadir + r"\train_label_" + str(i) + ".csv") as csvfile:
		reader = csv.reader(csvfile)
		temp = [[int(j) for j in i] for i in reader] 
	tsL += random.sample(temp, tsdataextract[i])
tsL = Dataset(tsL)

#computation
variance = trL.get_var()
print("variance =", variance)
trcl = treelet_classifier(trL, kernel("ra", [np.sqrt(variance)]), labeling(trdataextract))
print("start: build")
trcl.build()
print("start: predict")
L = trcl.predict_multiple(tsL)
realL = labeling(tsdataextract)
print(L)
print(realL)
print(trcl.training_error())
print(sum([L[i] == realL[i] for i in range(len(tsL))]) / len(tsL))
