from SA_method import SA_clust
import qfs


class SA2_clust(SA_clust):
	def __init__ (self, dataset_ref, kernel, sample_para,  inner_sample_para=False, slice=False, num_clust=0, all_kernel=False):
		layer1_clust = SA_clust(dataset_ref, kernel, sample_para, slice, num_clust, all_kernel)
		layer1_clust.build()
		layer1_cluster = layer1_clust.clusters
		refined_samp = []
		inner_sample_para = inner_sample_para if inner_sample_para else sample_para
		for one_clust in layer1_cluster:
			if len(layer1_cluster[one_clust]) <= 1:
				continue
			layer2_clust = SA_clust(dataset_ref, kernel, inner_sample_para, layer1_cluster[one_clust], num_clust, all_kernel)
			layer2_clust.build()
			fine_clusters = layer2_clust.clusters
			for i in fine_clusters:
				if len(fine_clusters[i]) > 1:
					refined_samp.append(fine_clusters[i][0])
		super().__init__(dataset_ref, kernel, refined_samp, slice, num_clust, all_kernel)

