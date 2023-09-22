import logging

import networkx as nx
import sklearn.cluster
import torch
from sklearn.cluster import KMeans

from variational_distributions.observational_variational_distribution import qPsi
from variational_distributions.var_dists import qC, qZ, qPi, qCMultiChrom, qEpsilonMulti, qMuTau


class SplitAndMergeOperations:

    def __init__(self, cluster_split_threshold=0.01):
        self.cluster_split_threshold = cluster_split_threshold

    def split(self, method, obs, qc: qCMultiChrom | qC, qz: qZ, qpsi: qPsi, qpi: qPi, qeps:qEpsilonMulti=None,
              tree_list=None, tree_weights_list=None):
        if method == 'naive':
            self.naive_split(obs, qc, qz, qpsi, qpi)
        elif method == 'categorical':
            self.categorical_split(obs, qc, qz, qpsi, qpi)
        elif method == 'inlier':
            self.inlier_split(obs, qc, qz, qpsi, qpi, qeps, tree_list, tree_weights_list)

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
            qc.eta2[k_to_cluster] = qc.eta2[k_to_cluster] - torch.logsumexp(qc.eta2[k_to_cluster], dim=-1, keepdim=True)
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

    def get_cell_likelihoods(self, y, qc, qz, qmt, k):
        #import matplotlib
        #matplotlib.use('module://backend_interagg')
        #import matplotlib.pyplot as plt
        N, K = qz.pi.shape
        K, M_not_nan, A = qc.single_filtering_probs.shape
        not_nan_idx = ~torch.any(y.isnan(), dim=1)
        M_not_nan = not_nan_idx.sum()
        qz_pi_k = qz.pi[:, k]
        qz_k_idx = qz.pi.argmax(dim=-1) == k
        emission_means = torch.einsum('n, km -> kmn', qmt.nu, qc.single_filtering_probs.argmax(dim=-1)[:, not_nan_idx])
        Y_given_Z = torch.distributions.Normal(emission_means[k], qmt.exp_tau().expand(M_not_nan, N))
        log_p_y_mn_given_Z = Y_given_Z.log_prob(y[not_nan_idx, :])
        log_p_y_n_given_Z = log_p_y_mn_given_Z.sum(dim=0)
        log_p_y_n_qz_k = log_p_y_n_given_Z[qz_k_idx]
        #plt.plot(log_p_y_n_given_Z.sort()[0])
        #plt.title('All cells')
        #plt.plot(log_p_y_n_qz_k.sort()[0])
        #plt.title('Cells of clone k')
        return log_p_y_n_qz_k, qz_k_idx

    def update_on_outliers(self, from_cluster, to_cluster, obs, qc, qz, qpsi, qpi, qeps, tree_list, tree_weights_list):
        log_probs, selected_cells_idx = self.get_cell_likelihoods(obs, qc, qz, qpsi, from_cluster)
        clusters = KMeans(n_clusters=3, random_state=0).fit(log_probs.reshape(log_probs.shape[0], 1))
        inlier_cluster = torch.argmin(torch.tensor(clusters.cluster_centers_))
        inlier_cells_idx = torch.tensor([1 if i == inlier_cluster else 0 for i in clusters.labels_], dtype=torch.long)

        inlier_cells_idx_2 = torch.zeros_like(selected_cells_idx)
        j = 0
        for i in range(selected_cells_idx.shape[0]):
            if selected_cells_idx[i] == 1:
                inlier_cells_idx_2[i] = inlier_cells_idx[j]
                j += 1

        not_inlier_cells_idx = torch.tensor([1 if e == 0 else 0 for e in inlier_cells_idx_2], dtype=torch.long)
        N_temp = inlier_cells_idx.sum()
        # hard assign cells
        qz.pi[inlier_cells_idx_2, :] = 0.
        qz.pi[inlier_cells_idx_2, from_cluster] = 1.
        qz.pi[not_inlier_cells_idx, :] = 0.
        qz.pi[not_inlier_cells_idx, to_cluster] = 1.

        eta1, eta2 = qc.update_CAVI(obs, qeps, qz, qpsi, tree_list, tree_weights_list)
        qc.eta1[from_cluster] = eta1[from_cluster]
        qc.eta2[from_cluster] = eta2[from_cluster]
        qc.eta2[to_cluster] = eta2[to_cluster]
        qc.eta2[to_cluster] = eta2[to_cluster]

        qc.compute_filtering_probs()
        delta = qpi.update_CAVI(qz)
        qpi.concentration_param[to_cluster] = delta[to_cluster]
        qpi.concentration_param[from_cluster] = delta[from_cluster]
        return 0
        config_temp = qc.config
        config_temp.n_nodes = 2
        config_temp.n_cells = N_temp
        if type(qc) == qCMultiChrom:
            qc_temp = qCMultiChrom(config_temp)
        else:
            qc_temp = qC(config_temp)
        qz_temp = qZ(config_temp)
        qz_temp.pi[:, 0] = 0.
        qz_temp.pi[:, 1] = 1.
        q_eps_temp = qEpsilonMulti(config_temp)
        q_eps_temp.initialize(method='non-mutation')
        q_psi_temp = qMuTau(config_temp)

        tree = nx.DiGraph()
        tree.add_edge(0, 1)
        qc_temp.initialize()
        eta1, eta2 = qc_temp.update_CAVI(obs, q_eps_temp, qz_temp, q_psi_temp)
        qc.eta1[to_cluster] = eta1
        qc.eta2[to_cluster] = eta2

    def inlier_split(self, obs, qc, qz, qpsi, qpi, qeps, tree_list, tree_weights_list):
        """
       Implements the split part of the split algorithm commonly used in Expectation Maximization.
       Splits a cluster k1, selected using split-from-selection-strategy, to an empty cluster, k2, by finding the set of
       cells assigned to k1 with highest likelihood (inlier cells), then updating k1 only on this set of cells and
       updating k2 only on the remaining cells.
       """
        # Select clusters to reassign
        cluster_assignments_avg, empty_clusters = self.find_empty_clusters(qz)
        if empty_clusters.shape[0] == 0:
            logging.debug(f'No empty clusters found')
            return False

        into_cluster, from_cluster = self.select_clusters_to_split_categorical(
            cluster_assignments_avg, empty_clusters)

        self.update_on_outliers(from_cluster, into_cluster, obs, qc, qz, qpsi, qpi, qeps, tree_list, tree_weights_list)

        return True