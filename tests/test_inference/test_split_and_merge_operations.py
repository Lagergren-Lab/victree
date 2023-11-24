import copy
import unittest

import networkx as nx
import numpy as np
import torch

from inference.split_and_merge_operations import SplitAndMergeOperations
from inference.victree import VICTree
from tests import utils_testing
from utils.config import Config
from variational_distributions.joint_dists import FixedTreeJointDist
from variational_distributions.var_dists import qZ, qCMultiChrom, qEpsilonMulti, qPi, qMuTau, qC


class SplitAndMergeOperationsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.cluster_split_threshold = 0.01
        self.split_and_merge_op = SplitAndMergeOperations(cluster_split_threshold=self.cluster_split_threshold)

    def test_merge(self):
        N = 10
        M = 20
        K = 4
        A = 5

        config = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M, chromosome_indexes=[5, 10], eps0=(A-1)/A)
        qc = qC(config)
        qeps = qEpsilonMulti(config)
        qz = qZ(config)
        qpi = qPi(config)
        qmt = qMuTau(config)
        obs = torch.ones(M, N)
        T = nx.DiGraph()
        T.add_edge(0, 1)
        T.add_edge(1, 2)
        T.add_edge(0, 3)
        q = FixedTreeJointDist(obs, config, qc, qz, qeps, qmt, qpi, T)
        qc.initialize(method='clonal', obs=obs)
        qc.eta1[3, :] = torch.rand(A)
        qc.eta2[3, :, :, :] = torch.rand(M-1, A, A)
        qc.compute_filtering_probs(3)
        q.pi.initialize(method='uniform')
        q.mt.initialize(method='prior')
        q.eps.initialize(method='uniform')
        q.z.update(qpsi=qmt, qc=qc, qpi=qpi, obs=obs)
        pre_merge_probs = copy.deepcopy(q.z.pi[:, 1])
        self.assertTrue(torch.allclose(q.z.pi[:, 1], q.z.pi[:, 2]))
        self.split_and_merge_op.merge(obs=obs, q=q, trees=[T], tree_weights=[1.0])

        self.assertTrue(torch.all(q.z.pi[:, 1] > pre_merge_probs))
        self.assertTrue(torch.all(q.z.pi[:, 2] == 0.))

    def test_update_assignment_probabilities(self):
        pass

    def test_update_cluster_concentration_parameters(self):
        pass

    def test_update_cluster_profiles(self):
        pass

    def test_select_clusters_to_split_by_largest_cluster(self):
        cluster_assignments_avg = torch.tensor([0.2, 0.005, 0.5, 0.295])
        empty_clusters = [1]
        k_merge_cluster, k_split_cluster, largest_clusters_idx = \
            self.split_and_merge_op.select_clusters_to_split_by_largest_cluster(
                cluster_assignments_avg, empty_clusters)
        self.assertEqual(k_merge_cluster, 1)
        self.assertEqual(k_split_cluster, 2)
        self.assertTrue(torch.equal(largest_clusters_idx, torch.argsort(cluster_assignments_avg, descending=True)))

    def test_select_clusters_to_split_categorical(self):
        cluster_assignments_avg = torch.tensor([0.2, 0.005, 0.5, 0.295])
        empty_clusters = [1]
        n_iter = 1000
        into_cluster_list = np.zeros((n_iter,))
        from_cluster_list = np.zeros((n_iter,))

        for i in range(n_iter):
            into_cluster, from_cluster = \
                self.split_and_merge_op.select_clusters_to_split_categorical(
                    cluster_assignments_avg, empty_clusters)
            into_cluster_list[i] = into_cluster
            from_cluster_list[i] = from_cluster

        n_samples_0 = np.where(from_cluster_list == 0)[0].sum()
        n_samples_1 = np.where(from_cluster_list == 1)[0].sum()
        n_samples_2 = np.where(from_cluster_list == 2)[0].sum()
        n_samples_3 = np.where(from_cluster_list == 3)[0].sum()
        self.assertTrue(n_samples_2 > n_samples_3)
        self.assertTrue(n_samples_3 > n_samples_0)
        self.assertTrue(n_samples_0 > n_samples_1)

    def test_find_empty_clusters(self):
        config = Config()
        qz = qZ(config)
        k_empty_1 = 1
        k_empty_2 = 4
        cluster_assignments_probs = torch.ones(config.n_cells, config.n_nodes)
        cluster_assignments_probs[:, k_empty_1] = 0
        cluster_assignments_probs[:, k_empty_2] = 0
        cluster_assignments_probs = cluster_assignments_probs / cluster_assignments_probs.sum(dim=-1, keepdims=True)
        qz.pi = cluster_assignments_probs
        cluster_assignments_avg, empty_clusters = self.split_and_merge_op.find_empty_clusters(qz=qz)
        for e in empty_clusters:
            self.assertTrue(e in [k_empty_1, k_empty_2], msg=f'{e} found but expeted {[k_empty_1, k_empty_2]}')

    def test_max_ELBO_split(self):
        # Check all available clones.

        # Simulate K clusters
        # Assign q
        # Assign cells
        K = 5
        tree = utils_testing.get_tree_K_nodes_random(K)
        M = 1000
        N = 50
        A = 7
        delta0 = 3.
        y, c, z, pi, mu, tau, eps, eps0, chr_idx = utils_testing.simulate_full_dataset_no_pyro(N, M, A, tree,
                                                                                               dir_alpha0=delta0,
                                                                                               cne_length_factor=5,
                                                                                               return_chr_idx=True)
        config = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M, step_size=1.0, split='ELBO',
                        chromosome_indexes=chr_idx)
        qc = qCMultiChrom(config)
        qeps = qEpsilonMulti(config)
        qz = qZ(config)
        qpi = qPi(config)
        qmt = qMuTau(config)

        q = FixedTreeJointDist(y, config, qc, qz, qeps, qmt, qpi, tree)
        q.initialize()
        qmt.initialize(method='prior')
        utils_testing.initialize_qc_to_true_values(c, A, qc)
        qz.update(qmt, qc, qpi, y)
        # Simulate degeneracy of clones 1 and '
        empty_clone_idx = 2
        absorbing_clone_idx = 1
        qz.pi[:, empty_clone_idx] = 0.
        qz.pi[:, absorbing_clone_idx] = 1.
        qz.pi = qz.pi / torch.sum(qz.pi, dim=1, keepdim=True)

        victree = VICTree(config, q, y, draft=True)

        elbo_pre_split = victree.compute_elbo()
        victree.split()

        elbo_after_split = victree.compute_elbo()

        self.assertGreater(elbo_after_split, elbo_pre_split, msg="ELBO decreased after split.")
