import logging

import torch

from variational_distributions.observational_variational_distribution import qPsi
from variational_distributions.var_dists import qC, qZ, qPi, qCMultiChrom


class SplitAndMergeOperations:

    def __init__(self, cluster_split_threshold=0.01):
        self.cluster_split_threshold = cluster_split_threshold

    def split(self, obs, qc: qCMultiChrom | qC, qz: qZ, qpsi: qPsi, qpi: qPi):
        """
        Implements the split part of the split-and-merge algorithm commonly used in Expectation Maximization.
        Splits a cluster k1, selected using split-from-selection-strategy, to an empty cluster, k2, by copying over the
        copy number profile of k1 to k2 with some added noise, redistributing the concentration parameters equally and
        cluster assignment for cells assigned to k1 by the qZ CAVI update.
        """
        # Select clusters to reassign
        cluster_assignments_avg, empty_clusters = self.find_empty_clusters(qz)
        if empty_clusters.shape[0] == 0:
            logging.debug(f'No empty clusters found')
            return False

        k_merge_cluster, k_split_cluster, largest_clusters_idx = self.select_clusters_to_split(cluster_assignments_avg,
                                                                                               empty_clusters)
        # perturbate copy number profile
        self.update_cluster_profiles(qc, k_merge_cluster, k_split_cluster)

        # Set concentration parameters equal
        self.update_cluster_concentration_parameters(qpi, k_merge_cluster, k_split_cluster)

        # Manually update assignments i.e. reassign cells from the large cluster to the empty
        # Select cells to update
        selected_cells = qz.pi.argmax(dim=-1) == largest_clusters_idx[0]

        # Calculate new assignment probability of selected cells using CAVI update
        self.update_assignment_probabilities(obs, qc, qpi, qpsi, qz, selected_cells)
        return True

    def update_assignment_probabilities(self, obs, qc, qpi, qpsi, qz, selected_cells):
        assignments = qz.update_CAVI(qpsi, qc, qpi, obs)
        qz.pi[selected_cells, :] = assignments[selected_cells, :]

    def update_cluster_concentration_parameters(self, qpi: qPi, k_merge_cluster, k_split_cluster):
        qpi.concentration_param[k_merge_cluster] = qpi.concentration_param[k_split_cluster] / 2
        qpi.concentration_param[k_split_cluster] = qpi.concentration_param[k_split_cluster] / 2

    def update_cluster_profiles(self, qc: qCMultiChrom | qC, k_merge_cluster, k_split_cluster):
        if type(qc) == qC:
            qc.eta1[k_merge_cluster] = qc.eta1[k_split_cluster] + 0.05 * torch.randn(qc.config.n_states)
            qc.eta2[k_merge_cluster] = qc.eta2[k_split_cluster] + \
                                       0.05 * torch.randn((qc.config.chain_length - 1, qc.config.n_states,
                                                           qc.config.n_states))
        else:
            for qc_chrom in qc.qC_list:
                qc_chrom.eta1[k_merge_cluster] = qc_chrom.eta1[k_split_cluster] + 0.05 * torch.randn(qc_chrom.config.n_states)
                qc_chrom.eta2[k_merge_cluster] = qc_chrom.eta2[k_split_cluster] + \
                                                 0.05 * torch.randn((qc_chrom.config.chain_length - 1, qc_chrom.config.n_states,
                                                                     qc_chrom.config.n_states))
        qc.compute_filtering_probs()

    def select_clusters_to_split(self, cluster_assignments_avg, empty_clusters, root=0):
        # Select clusters to split
        largest_clusters_values, largest_clusters_idx = torch.sort(cluster_assignments_avg, descending=True)

        cumulative_cluster_prob = torch.cumsum(largest_clusters_values, dim=0)
        print(f'Split clusters with indexes: {empty_clusters}')
        logging.debug(f'Based on average cluster assignments: {cluster_assignments_avg}')
        # Split clusters into empty clusters (duplication)
        # naive split - split largest cluster into first empty cluster
        logging.debug(f'')
        k_split_cluster = largest_clusters_idx[0]
        k_merge_cluster = empty_clusters[0]
        return k_merge_cluster, k_split_cluster, largest_clusters_idx

    def find_empty_clusters(self, qz: qZ, root=0):
        cluster_assignments_avg = qz.pi.mean(dim=0)
        empty_clusters = torch.where(cluster_assignments_avg < self.cluster_split_threshold)[0]
        if root in empty_clusters:
            empty_clusters = empty_clusters[empty_clusters != 0]
        return cluster_assignments_avg, empty_clusters
