import csv
import os
import random

import numpy as np

from Dataset import Dataset
from kernel import kernel
from treelet_classifier import treelet_classifier, MajorityVote
from SA2_method import SA2_clust
from cluster_classification_mix import cluster_classification_mix

os.chdir(os.path.dirname(os.path.realpath(__file__)))
datadir = r"C:\D\senior_thesis\handwritten_num\samples\comp20data"
trdataextract = {i:200 for i in range(10)}
tsdataextract = {i:100 for i in range(10)}
	
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

trlab = labeling(trdataextract)
tslab = labeling(tsdataextract)

#computation
variance = trL.get_var()
print("variance =", variance)
ker = kernel("ra", [np.sqrt(variance)])
sacl = SA2_clust(trL, ker, sample_para=300, inner_sample_para=10, num_clust=100)
trcl = cluster_classification_mix(trL, trlab, Clust_method=sacl, Classify_class=MajorityVote)
print("start: build")
trcl.build()
print("start: predict")
L = trcl.predict_multiple(tsL)

print(L)
print(tslab)
print(trcl.training_error())
print(sum([L[i] == tslab[i] for i in range(len(tsL))]) / len(tsL))
