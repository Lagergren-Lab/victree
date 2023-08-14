import unittest

import numpy as np
import torch

from tests import utils_testing
from utils.config import Config
from utils.tree_utils import tree_to_newick
from variational_distributions.var_dists import qEpsilonMulti, qT, qC, qEpsilon


class TreeMetricsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.qt5 = qT(config=Config(n_nodes=5))

    def test_q_T_running_for_two_simple_Ts_random_qC(self):
        K = 5
        utils_testing.get_tree_K_nodes_random(K)


    def test_tree_enumeration(self):
        trees, trees_log_prob = self.qt5.enumerate_trees()

        self.assertAlmostEqual(trees_log_prob.exp().sum().item(), 1., places=5,
                               msg='q(T) is not normalized')

    def test_tree_sampling_accuracy(self):
        trees, trees_log_prob = self.qt5.enumerate_trees()
        self.qt5.initialize()
        dsl_trees, dsl_trees_log_weights = self.qt5.get_trees_sample(sample_size=1000,
                                                                     torch_tensor=True, log_scale=True)
        self.assertAlmostEqual(dsl_trees_log_weights.exp().sum().item(), 1., places=5)
        unique_dsl_trees = {}
        for t, lw in zip(dsl_trees, dsl_trees_log_weights):
            t_str = tree_to_newick(t)
            if t_str not in unique_dsl_trees:
                unique_dsl_trees[t_str] = lw
            else:
                unique_dsl_trees[t_str] = np.logaddexp(unique_dsl_trees[t_str], lw)

        k = 10
        # sort enumerated trees
        _, topk_idx = torch.topk(trees_log_prob, k=k)
        print(f"TOP {k} enumerated trees")
        for i in topk_idx:
            print(f"{tree_to_newick(trees[i])}: {trees_log_prob[i].exp()}")

        # sort unique sampled trees
        topk_sampled_str = [None] * k
        sampled_lw = torch.zeros(k) - torch.inf
        for i in range(k):
            for t_str, lw in unique_dsl_trees.items():
                if lw > sampled_lw[i]:
                    sampled_lw[i] = lw
                    topk_sampled_str[i] = t_str

            unique_dsl_trees.pop(topk_sampled_str[i])

        print(f"TOP {k} sampled trees")
        for t_str, lw in zip(topk_sampled_str, sampled_lw):
            print(f"{t_str}: {lw.exp()}")



