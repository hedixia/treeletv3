import treelet
from Dataset import Dataset
from treelet_clust import treelet_clust
from SA_method import SA_clust
from SACSA_method import SACSA_clust
import numpy as np
import csv 
import os 
from kernel import kernel 
import random 
import matplotlib.pyplot as plt

x = 10
dataset = Dataset([[i//10, i//10] for i in range(10*x)])
ker = kernel("ra", [1])
temp_sac = SA_clust(dataset, ker, 10*x, range(5*x), 10)
temp_sac.build()
print(temp_sac.sample_index)
temp_dl = temp_sac.get("C")
print(temp_dl)