import logging

import torch

from variational_distributions.observational_variational_distribution import qPsi
from variational_distributions.var_dists import qC, qZ, qPi, qCMultiChrom


class SplitAndMergeOperations:

    def __init__(self, cluster_split_threshold=0.01):
        self.cluster_split_threshold = cluster_split_threshold

    def split(self, method, obs, qc: qCMultiChrom | qC, qz: qZ, qpsi: qPsi, qpi: qPi):
        if method == 'naive':
            self.naive_split(obs, qc, qz, qpsi, qpi)
        elif method == 'categorical':
            self.categorical_split(obs, qc, qz, qpsi, qpi)

    def naive_split(self, obs, qc: qCMultiChrom | qC, qz: qZ, qpsi: qPsi, qpi: qPi):
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

        k_merge_cluster, k_split_cluster, largest_clusters_idx = self.select_clusters_to_split_by_largest_cluster(
            cluster_assignments_avg, empty_clusters)
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

    def update_cluster_profiles(self, qc: qCMultiChrom | qC, k_to_cluster, k_split_cluster):
        if type(qc) == qC:
            qc.eta1[k_to_cluster] = qc.eta1[k_split_cluster] + 0.05 * torch.randn(qc.config.n_states)
            qc.eta1[k_to_cluster] = qc.eta1[k_to_cluster] - torch.logsumexp(qc.eta1[k_to_cluster], dim=-1)
            qc.eta2[k_to_cluster] = qc.eta2[k_split_cluster] + \
                                    0.05 * torch.randn((qc.config.chain_length - 1, qc.config.n_states,
                                                           qc.config.n_states))
            qc.eta2[k_to_cluster] = qc.eta2[k_to_cluster] - torch.logsumexp(qc.eta2[k_to_cluster], dim=-1)
        else:
            for qc_i in qc.qC_list:
                qc_i.eta1[k_to_cluster] = qc_i.eta1[k_split_cluster] + 0.05 * torch.randn(qc_i.config.n_states)
                qc_i.eta1[k_to_cluster] = qc_i.eta1[k_to_cluster] - torch.logsumexp(qc_i.eta1[k_to_cluster], dim=-1)
                qc_i.eta2[k_to_cluster] = qc_i.eta2[k_split_cluster] + \
                                          0.05 * torch.randn((qc_i.config.chain_length - 1, qc_i.config.n_states,
                                                                 qc_i.config.n_states))
                qc_i.eta2[k_to_cluster] = qc_i.eta2[k_to_cluster] - torch.logsumexp(qc_i.eta2[k_to_cluster], dim=-1,
                                                                                    keepdim=True)

        qc.compute_filtering_probs()

    def select_clusters_to_split_by_largest_cluster(self, cluster_assignments_avg, empty_clusters, root=0):
        # Select from cluster by largest cell to clone assignment
        largest_clusters_values, largest_clusters_idx = torch.sort(cluster_assignments_avg, descending=True)
        from_cluster_idx = largest_clusters_idx[0]
        to_cluster_idx = empty_clusters[0]
        logging.debug(f'Split from cluster {from_cluster_idx} into cluster {to_cluster_idx} by argmax of average'
                      f' cluster assignments: {cluster_assignments_avg}')
        return to_cluster_idx, from_cluster_idx, largest_clusters_idx

    def select_clusters_to_split_categorical(self, cluster_assignments_avg, empty_clusters, root=0):
        categorical_rv = torch.distributions.categorical.Categorical(probs=cluster_assignments_avg)
        from_cluster_idx = categorical_rv.sample()
        to_cluster_idx = empty_clusters[0]
        logging.debug(f'Split from cluster {from_cluster_idx} into cluster {to_cluster_idx} by categorical sample'
                      f' from average cluster assignments: {cluster_assignments_avg}')
        return to_cluster_idx, from_cluster_idx

    def find_empty_clusters(self, qz: qZ, root=0):
        cluster_assignments_avg = qz.pi.mean(dim=0)
        empty_clusters = torch.where(cluster_assignments_avg < self.cluster_split_threshold)[0]
        if root in empty_clusters:
            empty_clusters = empty_clusters[empty_clusters != 0]
        return cluster_assignments_avg, empty_clusters

    def categorical_split(self, obs, qc: qCMultiChrom | qC, qz: qZ, qpsi: qPsi, qpi: qPi):
        """
        Implements the split part of the split algorithm commonly used in Expectation Maximization.
        Splits a cluster k1, selected using split-from-selection-strategy, to an empty cluster, k2, by copying over the
        copy number profile of k1 to k2 with some added noise, redistributing the concentration parameters equally and
        cluster assignment for cells assigned to k1 by the qZ CAVI update.
        """
        # Select clusters to reassign
        cluster_assignments_avg, empty_clusters = self.find_empty_clusters(qz)
        if empty_clusters.shape[0] == 0:
            logging.debug(f'No empty clusters found')
            return False

        into_cluster, from_cluster = self.select_clusters_to_split_categorical(
            cluster_assignments_avg, empty_clusters)
        # perturbate copy number profile
        self.update_cluster_profiles(qc, into_cluster, from_cluster)

        # Set concentration parameters equal
        self.update_cluster_concentration_parameters(qpi, into_cluster, from_cluster)

        # Manually update assignments i.e. reassign cells from the large cluster to the empty
        # Select cells to update
        selected_cells = qz.pi.argmax(dim=-1) == from_cluster

        # Calculate new assignment probability of selected cells using CAVI update
        self.update_assignment_probabilities(obs, qc, qpi, qpsi, qz, selected_cells)
        return True
