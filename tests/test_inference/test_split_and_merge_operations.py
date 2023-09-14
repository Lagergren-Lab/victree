import unittest

import numpy as np
import torch

from inference.split_and_merge_operations import SplitAndMergeOperations
from utils.config import Config
from variational_distributions.var_dists import qZ


class SplitAndMergeOperationsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.cluster_split_threshold = 0.01
        self.split_and_merge_op = SplitAndMergeOperations(cluster_split_threshold=self.cluster_split_threshold)

    def test_split(self):
        pass

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
