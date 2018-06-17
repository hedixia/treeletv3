import csv
import os
import random

import numpy as np

from Dataset import Dataset
from kernel import kernel
from treelet_classifier import treelet_classifier, MajorityVote
from SA2_method import SA2_clust
from cluster_classification_mix import cluster_classification_mix
from dataset_loader import loadDataset


os.chdir(os.path.dirname(os.path.realpath(__file__)))
datadir = r"C:\D\senior_thesis\handwritten_num\samples\comp20data"
training_percentage = 0.5
recnum = 1

#load and split data
alldat = loadDataset(datadir, ["dat.csv"], datatype=float, labcol=0)
n = len(alldat)
trn = int(n * training_percentage)
tsn = n - trn
trslice = random.sample(range(n), trn)
tsslice = [i for i in range(n) if i not in trslice]
trdat = alldat.getSlice(trslice)
tsdat = alldat.getSlice(tsslice)

#computation
variance = [trdat.Var, tsdat.Var]
print("var: ", variance)
ker = kernel("ra", [np.sqrt(variance[0])])
trc = treelet_classifier(trdat, ker, trdat.label)
trc.build()
prediction_lab = trc.predict_multiple(tsdat)
print("training error: ", trc.training_error())
testerror = sum([prediction_lab[i] != tsdat.label[i] for i in range(tsn)]) / tsn
print("test error: ", testerror)

#write test to file
writeL = [
	"Test No." + str(recnum).zfill(3),
	"Training Error: " + str(trc.training_error()),
	"Test Error: " + str(testerror),
	"Training data size: " + str(trn),
	"Test data size: " + str(tsn),
	"Training data slice: " + str(trslice)
]

openfile = open ("record" + str(recnum).zfill(3) + ".txt", "w")
openfile.writelines(writeL)
openfile.close()