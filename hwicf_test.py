import csv
import os
import random

import numpy as np

from Dataset import Dataset
from kernel import kernel
from treelet_classifier import treelet_classifier, MajorityVote
from SA2_method import SA2_clust
from cluster_classification_mix import cluster_classification_mix
from SVM import SVM

os.chdir(os.path.dirname(os.path.realpath(__file__)))
datadir = r"C:\D\senior_thesis\handwritten_num\samples\comp20data"
iter_num = 3
trdataextract = {i:1000 for i in range(10)}
tsdataextract = {i:100 for i in range(10)}

def labeling (idict):
	L = []
	for i in range(10):
		if i not in idict:
			continue
		L += [i] * idict[i]
	return L

#fetch training data
trLs = [[] for i in range(iter_num)]
for i in range(10):
	if i not in trdataextract:
		continue
	with open(datadir + r"\train_label_" + str(i) + ".csv") as csvfile:
		reader = csv.reader(csvfile)
		temp = [[int(j) for j in i] for i in reader]
	for trL in trLs:
		trL += random.sample(temp, trdataextract[i])
trLs = [Dataset(trL) for trL in trLs]
del trL

#fetch test data
tsL = []
for i in range(10):
	if i not in tsdataextract:
		continue
	with open(datadir + r"\test_label_" + str(i) + ".csv") as csvfile:
		reader = csv.reader(csvfile)
		temp = [[int(j) for j in i] for i in reader] 
	tsL += random.sample(temp, tsdataextract[i])
tsL = Dataset(tsL)

trlab = labeling(trdataextract)
tslab = labeling(tsdataextract)

#computation
variance = [trL.get_var() for trL in trLs]
print("variance =", variance)
ker = kernel("ra", [1])
"""
sacl = SA2_clust(trL, ker, sample_para=300, inner_sample_para=10, num_clust=20)
cfc = treelet_classifier
cck = {"kernel":ker, "CLM":MajorityVote, "all_kernel":False}
trc = cluster_classification_mix(trL, trlab, Clust_method=sacl,
                                  Classify_class=cfc, Classify_class_kwargs=cck)
print("start: build")
trc.build()
print("start: predict")
Ls.append(trc.predict_multiple(tsL))
"""

Ls = []
for trL in trLs:
	trc = treelet_classifier(trL, ker, trlab)
	print("start: build")
	trc.build()
	print("start: predict")
	Ls.append(trc.predict_multiple(tsL))

rL = {}
for i in set(Ls[0]):
	lst = [L[i] for L in Ls]
	rL[i] = max(set(lst), key=lst.count)

print(trc.training_error())
cpt = sum([rL[i] == tslab[i] for i in range(len(tsL))]) / len(tsL)
print(cpt)
temp = np.zeros((10,10))
for i in rL:
	temp[rL[i]][tslab[i]] += 1
print(temp)

#write test to file
openfile = open ("record012.txt", "w")
openfile.write("number of iteration: " + str(iter_num) + "\n")
openfile.write("training error: " + str(trc.training_error()) + "\n")
openfile.write("Correct percentage: " + str(cpt) + "\n")
openfile.write(temp.__str__())