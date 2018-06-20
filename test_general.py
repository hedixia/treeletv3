import os
import random

import numpy as np

from dataset_loader import loadDataset
from kernel import kernel
from treelet_classifier import treelet_classifier

os.chdir(os.path.dirname(os.path.realpath(__file__)))
datadir = r"C:\Users\Hedi Xia\Desktop\datasets\ADD\\"
training_percentage = 0.6
recnum = 4

# load and split data
alldat = loadDataset(datadir, ["dat.csv"], datatype=float, labcol=0)
n = len(alldat)
print(n)
trn = int(n * training_percentage)
tsn = n - trn
trslice = random.sample(range(n), trn)
tsslice = [i for i in range(n) if i not in trslice]
trdat = alldat.getSlice(trslice)
tsdat = alldat.getSlice(tsslice)

# computation
variance = [trdat.Var, tsdat.Var]
print("var: ", variance)
ker = kernel("ra", [np.sqrt(variance[0])])
trc = treelet_classifier(trdat, ker, trdat.label)
trc.build()
prediction_lab = trc.predict_multiple(tsdat)
print("training error: ", trc.training_error())
testerror = sum([prediction_lab[i] != tsdat.label[i] for i in range(tsn)]) / tsn
print("test error: ", testerror)

# write test to file
writeL = [
	"Test No." + str(recnum).zfill(3) + "\n",
	"Data dir: " + datadir[-5:-2] + "\n",
	"Training Error: " + str(trc.training_error()) + "\n",
	"Test Error: " + str(testerror) + "\n",
	"Training data size: " + str(trn) + "\n",
	"Test data size: " + str(tsn) + "\n",
	"Training data percent: " + str(training_percentage) + "\n",
	"Training data slice: " + str(sorted(trslice)) + "\n"
]

openfile = open("record" + str(recnum).zfill(3) + ".txt", "w")
openfile.writelines(writeL)
openfile.close()
