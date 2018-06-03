import treelet
from treelet_clust import treelet_clust
from treelet_classifier import treelet_classifier
import numpy as np
import csv 
import os 
from kernel import kernel 
import random 
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.realpath(__file__)))
datadir = r"C:\D\senior_thesis\handwritten_num\samples\comp20data"
trdataextract = {0:100, 1:100, 2:100, 3:100, 4:100, 5:100, 6:100, 7:100, 8:100, 9:100}
tsdataextract = {0:100, 1:100, 2:100, 3:100, 4:100, 5:100, 6:100, 7:100, 8:100, 9:100}
coi = 1

def purity (dictlist):
	sum = 0
	totalmaxsum = 0
	for k in dictlist:
		max = 0 
		for i in dictlist[k]:
			sum += i 
			if i > max:
				max = i 
		totalmaxsum += max 
	return totalmaxsum / sum 
	
def labeling (idict):
	L = []
	for i in range(10):
		if i not in idict:
			continue
		for j in range(idict[i]):
			L.append(i)
	return L

trL = []
for i in range(10):
	if i not in trdataextract:
		continue
	with open(datadir + r"\train_label_" + str(i) + ".csv") as csvfile:
		reader = csv.reader(csvfile)
		temp = [[int(j) for j in i] for i in reader] 
	trL += random.sample(temp, trdataextract[i])

tsL = []
for i in range(10):
	if i not in tsdataextract:
		continue
	with open(datadir + r"\train_label_" + str(i) + ".csv") as csvfile:
		reader = csv.reader(csvfile)
		temp = [[int(j) for j in i] for i in reader] 
	tsL += random.sample(temp, tsdataextract[i])
	
"""
#majority vote
def CLM(training_set, training_label, test_data):
	lkey = list(set(training_label))
	lval = [training_label.count(i) for i in lkey]
	temp = lkey[lval.index(max(lval))]
	return [temp for i in test_data]
"""

"""
#SVM
def CLM(training_set, training_label, test_data):
	if len(set(training_label)) == 1:
		return [training_label[0] for i in test_data]
	clf = svm.SVC(decision_function_shape='ovo')
	clf.fit(training_set, training_label)
	return clf.predict(np.array(test_data))
"""

trL = np.matrix(trL)
trcl = treelet_classifier(trL, kernel("ra", [coi]), labeling(trdataextract))
print("start: build")
trcl.build()
print("start: predict")
L = trcl.predict_multiple(tsL)
realL = labeling(tsdataextract)
print(np.mean(np.array(L) == np.array(realL)))
