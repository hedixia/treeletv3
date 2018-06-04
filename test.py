import os

from Dataset import Dataset
from SA2_method import SA2_clust
from SA_method import SA_clust
from kernel import kernel

os.chdir(os.path.dirname(__file__))

x = 10
dataset = Dataset([[i / 20, i // 10] for i in range(10 * x)])
ker = kernel("ra", [1])

temp_sac1 = SA_clust(dataset, ker, 10, range(100), 10, False)
temp_sac1.build()
temp_sac1.show("SA")

temp_sac2 = SA2_clust(dataset, ker, 10, 2, range(100), 10, False)
temp_sac2.build()
temp_sac2.show("SA2")
