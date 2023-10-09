import copy
import logging

import networkx as nx
import sklearn.cluster
import torch
from sklearn.cluster import KMeans

from variational_distributions.joint_dists import VarTreeJointDist, FixedTreeJointDist
from variational_distributions.observational_variational_distribution import qPsi
from variational_distributions.var_dists import qC, qZ, qPi, qCMultiChrom, qEpsilonMulti, qMuTau


class SplitAndMergeOperations:

    def __init__(self, cluster_split_threshold=0.01):
        self.cluster_split_threshold = cluster_split_threshold

    def split(self, method, obs, q: VarTreeJointDist | FixedTreeJointDist, tree_list=None, tree_weights_list=None):
        if method == 'naive':
            self.naive_split(obs, q)
        elif method == 'categorical':
            self.categorical_split(obs, q)
        elif method == 'ELBO':
            self.max_ELBO_split(obs, q, tree_list, tree_weights_list)
        elif method == 'inlier':
            self.inlier_split(obs, q, tree_list, tree_weights_list)

    def naive_split(self, obs, q: VarTreeJointDist | FixedTreeJointDist):
        """
        Implements the split part of the split-and-merge algorithm commonly used in Expectation Maximization.
        Splits a cluster k1, selected using split-from-selection-strategy, to an empty cluster, k2, by copying over the
        copy number profile of k1 to k2 with some added noise, redistributing the concentration parameters equally and
        cluster assignment for cells assigned to k1 by the qZ CAVI update.
        """
        qz = q.z
        qc = q.c
        qpi = q.pi
        qpsi = q.mt
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

    def update_cluster_concentration_parameters(self, qpi: qPi, split_into_cluster_idx, split_from_cluster_idx):
        qpi.concentration_param[split_into_cluster_idx] = qpi.concentration_param[split_from_cluster_idx] / 2
        qpi.concentration_param[split_from_cluster_idx] = qpi.concentration_param[split_from_cluster_idx] / 2

    def update_cluster_profiles(self, qc: qCMultiChrom | qC, k_to_cluster, k_split_cluster, obs=None):
        if type(qc) == qC:
            qc.eta1[k_to_cluster] = qc.eta1[k_split_cluster] + 0.05 * torch.randn(qc.config.n_states)
            qc.eta1[k_to_cluster] = qc.eta1[k_to_cluster] - torch.logsumexp(qc.eta1[k_to_cluster], dim=-1)
            qc.eta2[k_to_cluster] = qc.eta2[k_split_cluster]
            qc.eta2[k_to_cluster, :, :] = qc.eta2[k_split_cluster] + \
                                                0.05 * torch.randn((qc.config.n_states,
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

    def categorical_split(self, obs, q: VarTreeJointDist | FixedTreeJointDist):
        """
        Implements the split part of the split algorithm commonly used in Expectation Maximization.
        Splits a cluster k1, selected using split-from-selection-strategy, to an empty cluster, k2, by copying over the
        copy number profile of k1 to k2 with some added noise, redistributing the concentration parameters equally and
        cluster assignment for cells assigned to k1 by the qZ CAVI update.
        """
        qz = q.z
        qc = q.c
        qpi = q.pi
        qpsi = q.mt

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
        import matplotlib
        matplotlib.use('module://backend_interagg')
        import matplotlib.pyplot as plt
        N, K = qz.pi.shape
        K, M_not_nan, A = qc.single_filtering_probs.shape
        not_nan_idx = ~torch.any(y.isnan(), dim=1)
        M_not_nan = not_nan_idx.sum()
        qz_pi_k = qz.pi[:, k]
        qz_k_idx = qz.pi.argmax(dim=-1) == k
        emission_means = torch.einsum('n, km -> kmn', qmt.nu, qc.single_filtering_probs.argmax(dim=-1)[:, not_nan_idx])
        Y_given_Z = torch.distributions.Normal(emission_means[k], qmt.exp_tau().expand(M_not_nan, N))
        log_p_y_mn_given_Z = Y_given_Z.log_prob(y[not_nan_idx, :])
        log_p_y_mn_given_Z_k = log_p_y_mn_given_Z[:, qz_k_idx]
        log_p_y_n_given_Z = log_p_y_mn_given_Z.sum(dim=0)
        log_p_y_n_qz_k = log_p_y_n_given_Z[qz_k_idx]
        plt.plot(log_p_y_n_given_Z.sort()[0])
        plt.title('All cells')
        plt.show()
        plt.plot(log_p_y_n_qz_k.sort()[0])
        plt.title('Cells of clone k')
        plt.show()
        plt.plot(torch.std(log_p_y_mn_given_Z_k, dim=1))
        plt.show()
        return log_p_y_n_qz_k, qz_k_idx

    def update_on_outliers(self, from_cluster, to_cluster, obs, qc, qz, qpsi, qpi, qeps, tree_list, tree_weights_list):
        log_probs, selected_cells_idx = self.get_cell_likelihoods(obs, qc, qz, qpsi, from_cluster)
        clusters = KMeans(n_clusters=2, random_state=0).fit(log_probs.reshape(log_probs.shape[0], 1))
        inlier_cluster = torch.argmax(torch.tensor(clusters.cluster_centers_))
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

    def inlier_split(self, obs, q, tree_list, tree_weights_list):
        """
       Implements the split part of the split algorithm commonly used in Expectation Maximization.
       Splits a cluster k1, selected using split-from-selection-strategy, to an empty cluster, k2, by finding the set of
       cells assigned to k1 with highest likelihood (inlier cells), then updating k1 only on this set of cells and
       updating k2 only on the remaining cells.
       """
        qz = q.z
        qc = q.c
        qeps = q.eps
        qpsi = q.mt
        qpi = q.pi

        # Select clusters to reassign
        cluster_assignments_avg, empty_clusters = self.find_empty_clusters(qz)
        if empty_clusters.shape[0] == 0:
            logging.debug(f'No empty clusters found')
            return False

        into_cluster, from_cluster = self.select_clusters_to_split_categorical(
            cluster_assignments_avg, empty_clusters)

        self.update_on_outliers(from_cluster, into_cluster, obs, qc, qz, qpsi, qpi, qeps, tree_list, tree_weights_list)

        return True

    def find_clonal_bins(self, from_cluster, into_cluster, qc, qz, qmt, obs):
        raise Exception("Under Development.")
        import matplotlib
        matplotlib.use('module://backend_interagg')
        import matplotlib.pyplot as plt
        plt.plot(torch.std(obs, dim=1))
        plt.title("Std observations")
        plt.show()

        self.get_cell_likelihoods(obs, qc, qz, qmt, from_cluster)

    def max_ELBO_split(self, obs, q: VarTreeJointDist | FixedTreeJointDist, tree_list, tree_weights_list):
        """
        Implements the split part of the split algorithm commonly used in Expectation Maximization.
        Splits a cluster k1, selected using split-from-selection-strategy, to an empty cluster, k2, by copying over the
        copy number profile of k1 to k2 with some added noise, redistributing the concentration parameters equally and
        cluster assignment for cells assigned to k1 by the qZ CAVI update.
        """
        qz = q.z
        qc = q.c
        qeps = q.eps
        qpsi = q.mt
        qpi = q.pi
        # Select clusters to reassign
        cluster_assignments_avg, empty_clusters = self.find_empty_clusters(qz)
        if empty_clusters.shape[0] == 0:
            logging.debug(f'No empty clusters found')
            return False

        # for each candidate cluster, split cells in cluster and assign to new cluster. Update qC based on those cells
        # and calculate ELBO. Select split which maximizes the ELBO

        candidates_idxs = self.select_candidates_clusters_to_split_by_threshold(cluster_assignments_avg)
        elbos = []
        elbo_pre_split = q.compute_elbo() if type(q) is FixedTreeJointDist else q.compute_elbo(tree_list, tree_weights_list)
        logging.debug(f"ELBO before split: {elbo_pre_split}")
        N, M, K, A = (q.config.n_cells, q.config.chain_length, q.config.n_nodes, q.config.n_states)
        idx_empty_cluster = empty_clusters[0]

        eta_1_pre_split = copy.deepcopy(qc.eta1)
        eta_2_pre_split = copy.deepcopy(qc.eta2)
        best_elbo = -torch.inf
        for k in candidates_idxs:
            # Split cells of candidate cluster k into clusters i and j
            cells_in_k = torch.where(qz.pi.argmax(dim=1) == k)[0]
            N_k = len(cells_in_k)
            if N_k < 2:
                continue

            cells_in_i, cells_in_j = self.split_cells_by_observations(obs, cells_in_k)
            batch_i = torch.tensor(cells_in_i)
            batch_j = torch.tensor(cells_in_j)

            # Update qC on batches
            eta1_1, eta2_1 = qc.update_CAVI(obs[:, batch_i], qeps, qz, qpsi, tree_list, tree_weights_list, batch=batch_i)
            eta1_2, eta2_2 = qc.update_CAVI(obs[:, batch_j], qeps, qz, qpsi, tree_list, tree_weights_list, batch=batch_j)

            qc.set_params(eta1_1, eta2_1, idx_empty_cluster)
            #qc.eta1[idx_empty_cluster] = eta1_1[idx_empty_cluster]
            #qc.eta2[idx_empty_cluster] = eta2_1[idx_empty_cluster]
            qc.compute_filtering_probs(idx_empty_cluster)

            #qc.eta1[k] = eta1_2[k]
            #qc.eta2[k] = eta2_2[k]
            qc.set_params(eta1_1, eta2_1, k)
            qc.compute_filtering_probs(k)

            # Measure quality of split in terms of ELBO
            elbo_split = q.compute_elbo(tree_list, tree_weights_list) if type(q) is VarTreeJointDist else q.compute_elbo()
            elbos.append(elbo_split)

            logging.debug(f"ELBO of split candidate {k}: {elbo_split} ")

            if elbo_split > best_elbo:
                best_elbo = elbo_split
                best_eta1_1, best_eta2_1 = (eta1_1, eta2_1)
                best_eta1_2, best_eta2_2 = (eta1_2, eta2_2)
                best_cluster_idx = k
                best_batch_i = batch_i
                best_batch_j = batch_j

            # reset qc.eta1
            #qc.eta1[k] = eta_1_pre_split[k]
            #qc.eta2[k] = eta_2_pre_split[k]
            qc.set_params(eta_1_pre_split, eta_2_pre_split, [k])
            qc.compute_filtering_probs(k)

        # select highest elbo
        qc.set_params(best_eta1_1, best_eta2_1, best_cluster_idx)
        qc.set_params(best_eta1_2, best_eta2_2, best_cluster_idx, idx_empty_cluster)
        qc.compute_filtering_probs()

        #if elbos[selected_split_cluster_idx] < elbo_pre_split:
        #    logging.debug(f"No split found.")


        # Set concentration parameters equal
        self.update_cluster_concentration_parameters(qpi, idx_empty_cluster, best_cluster_idx)

        # Calculate new assignment probability of selected cells using CAVI update
        self.update_assignment_probabilities(obs, qc, qpi, qpsi, qz, torch.arange(0, N))
        return True

    def split_by_cell_qC_gradient(self, A, M, N_k, cells_in_k, k, obs, qc, qeps, qpsi, qz, tree_list,
                                  tree_weights_list):
        eta1_k = torch.zeros(N_k, A)
        eta2_k = torch.zeros(N_k, M - 1, A, A)
        n_batch_cells = 1
        for i in range(N_k):
            batch = cells_in_k[i:i + n_batch_cells]
            eta1_i, eta2_i = qc.update_CAVI(obs[:, batch], qeps, qz, qpsi, tree_list, tree_weights_list, batch=batch)
            eta1_k[i, :] = eta1_i[k, :]
            eta2_k[i, :, :, :] = eta2_i[k, :, :]
        eta_dist_matrix = self.calculate_euclidean_distances(eta2_k.exp())
        return eta_dist_matrix

    def select_candidates_clusters_to_split_from_bulk(self, cluster_assignments_avg):
        """
        Identifies the clones accounting for the bulk_fraction amount of cell to clone probability assignment and
        returns the indexes of these clones.
        """
        split_candidate_idxs = []
        bulk_fraction = 0.98
        K = cluster_assignments_avg.shape[0]
        cluster_assignments_avg_sorted, idxs = torch.sort(cluster_assignments_avg)
        cum_sum = cluster_assignments_avg_sorted[0]
        split_candidate_idxs.append(idxs[0])
        for k in range(1, K):
            if cum_sum > bulk_fraction:
                break
            cum_sum += cluster_assignments_avg_sorted[k]
            split_candidate_idxs.append(idxs[k])

        logging.debug(f'Candidate clusters to split: {split_candidate_idxs} based on avg assignments: '
                      f'{cluster_assignments_avg} accounting for {bulk_fraction} of assignment')
        return split_candidate_idxs

    def select_candidates_clusters_to_split_by_threshold(self, cluster_assignments_avg):
        """
        Returns the indexes of all clones with cell to clone assignment above 'threshold'.
        """
        threshold = 0.03
        split_candidate_idxs = torch.where(cluster_assignments_avg > threshold)[0]

        logging.debug(f'Candidate clusters to split: {split_candidate_idxs} based on avg assignments: '
                      f'{cluster_assignments_avg} with assignment threshold {threshold}')
        return split_candidate_idxs

    def calculate_euclidean_distances(self, eta2):
        n_vectors = eta2.shape[0]
        matrix = torch.zeros(n_vectors, n_vectors)
        for i in range(0, n_vectors):
            for j in range(i+1, n_vectors):
                matrix[i, j] = (eta2[i] - eta2[j]).pow(2).sum().sqrt()

        return matrix

    def split_cells_by_observations(self, obs, cell_idxs):
        not_nan_idx = ~torch.any(obs.isnan(), dim=1)
        obs_not_nan = obs[not_nan_idx, :]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(obs_not_nan[:, cell_idxs].T)
        labels = kmeans.labels_
        cells_in_cluster_0 = []
        cells_in_cluster_1 = []
        assert len(labels) == cell_idxs.shape[0]
        for i in range(len(labels)):
            if labels[i] == 0:
                cells_in_cluster_0.append(cell_idxs[i].item())
            else:
                cells_in_cluster_1.append(cell_idxs[i].item())
        return cells_in_cluster_0, cells_in_cluster_1

