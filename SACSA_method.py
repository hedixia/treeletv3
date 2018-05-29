from SA_method import SA_clust
import qfs


class SACSA_clust(SA_clust):
	def __init__ (self, dataset, slice, sample_para, num_clust, kernel):
		temp_clust = SA_clust(dataset, slice, sample_para)
		temp_clust.K = kernel
		temp_clust.tree()
		flc = temp_clust.clusters(num_clust, "D")
		refined_samp = []
		for clust_1 in flc:
			fine_SA_clust = SA_clust(dataset, flc[clust_1], sample_para)
			fine_SA_clust.K = kernel
			fine_SA_clust.tree()
			fine_clusters = fine_SA_clust.clusters(num_clust, "D")
			fine_samp = [flc[clust_1][k] for k in qfs.dlsamp(fine_clusters, 1)]
			refined_samp += fine_samp
		super().__init__(dataset, slice, refined_samp, kernel)

