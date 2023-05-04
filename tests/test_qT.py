import unittest

import numpy as np
import torch

from tests import utils_testing
from utils.config import Config, set_seed
from utils.tree_utils import tree_to_newick
from variational_distributions.var_dists import qEpsilonMulti, qT, qC, qEpsilon


class qTTestCase(unittest.TestCase):

    def setUp(self) -> None:
        set_seed(42)
        self.qt5: qT = qT(config=Config(n_nodes=5))

    def test_q_T_running_for_two_simple_Ts_random_qC(self):
        M = 20
        A = 5
        N = 5
        config = Config(n_nodes=N, n_states=A, chain_length=M)
        q_T = qT(config=config)
        T_list, q_C_pairwise_marginals = utils_testing.get_two_simple_trees_with_random_qCs(M, A, N)
        q_C = qC(config=config)
        q_C.couple_filtering_probs = q_C_pairwise_marginals
        q_epsilon = qEpsilonMulti(config=config)
        q_epsilon.initialize()
        # Act
        log_q_T = q_T.update_CAVI(T_list, q_C, q_epsilon)

        # Assert
        print(f"log_q_T of T_1 and T_2: {log_q_T}")

    def test_tree_enumeration(self):
        trees, trees_log_prob = self.qt5.enumerate_trees()

        self.assertAlmostEqual(trees_log_prob.exp().sum().item(), 1., places=5,
                               msg='q(T) is not normalized')

    def test_tree_sampling_accuracy(self):
        trees, trees_log_prob = self.qt5.enumerate_trees()
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

    def test_inference_fixed_ceps(self):
        # build ad-hoc c
        c = torch.tensor([
            [2] * 20,
            [3] * 10 + [2] * 10,
            [2] * 5 + [3] * 5 + [2] * 10,
            [2] * 3 + [3] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [2] * 5
        ])
        config = Config(4, n_states=7, eps0=1e-2, chain_length=c.shape[1],
                        wis_sample_size=100, step_size=.2, debug=True)

        qt = qT(config).initialize()
        qeps = qEpsilonMulti(config).initialize()
        qc = qC(config, true_params={'c': c})

        for i in range(10):
            qt.update(qc, qeps)
            t, w = qt.get_trees_sample()
            qeps.update(t, w, qc)

        # print(qc)
        # print(qt)
        # print(qeps)
        gt_tree_newick = "((2)1,3)0"
        tol = 3
        qt_pmf = qt.get_pmf_estimate()
        sorted_newick = sorted(qt_pmf, key=qt_pmf.get, reverse=True)
        self.assertTrue(gt_tree_newick in sorted_newick[:tol],
                        msg=f"true " + gt_tree_newick + f" not in the first {tol} trees. those are:"
                                                        f"{sorted_newick[0]} | {sorted_newick[1]} | {sorted_newick[2]}")




