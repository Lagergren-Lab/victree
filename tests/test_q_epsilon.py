import unittest

import networkx as nx
import torch

from utils.config import Config
from tests import utils_testing
from variational_distributions.var_dists import qEpsilon, qEpsilonMulti, qC


class qEpsilonTestCase(unittest.TestCase):

    def setUp(self) -> None:
        L = 5
        a = 1
        b = 1
        self.config = Config()
        # self.qeps = qEpsilon(self.config, a, b)
        self.qeps = qEpsilonMulti(self.config, a, b)

    def test_q_epsilon_running_for_two_simple_Ts_random_qC(self):
        # Arange
        M=20
        A=5
        T_list, q_C_pairwise_marginals = utils_testing.get_two_simple_trees_with_random_qCs(M, A)
        w_T = torch.tensor([0.3, 0.7])

        # Act
        a, b = self.qeps.update_CAVI(T_list, w_T, q_C_pairwise_marginals)

        # Assert
        print(f"Beta param a: {a}")
        print(f"Beta param b: {b}")

    def test_expectation_size(self):
        for u in range(self.config.n_nodes):
            for v in range(self.config.n_nodes):
                if u != v:
                    exp_zipping = self.qeps.exp_zipping((u,v))
                    self.assertEqual(exp_zipping.shape, (self.config.n_states,) * 4)

    def test_h_eps0(self):
        heps_0_marg = torch.sum(self.qeps.h_eps0(), dim=-1)
        self.assertTrue(torch.allclose(heps_0_marg,
                                       torch.ones(self.config.n_states)))

    def test_update(self):
        # design simple test: fix all other variables
        # and update the q_eps params
        cells_per_clone = 10
        mm = 10  # change this to increase length
        chain_length = mm * 10
        cfg = Config(n_nodes=3, n_states=5, n_cells=3 * cells_per_clone, chain_length=chain_length,
                     wis_sample_size=2, debug=True)
        qeps = qEpsilonMulti(cfg)
        qeps.initialize('uniform')

        # obs with 15 cells, 5 each to different clone
        # in order, clone 0, 1, 2
        cn_profile = torch.tensor(
            [[2] * 10 * mm,
             [2] * 4 * mm + [3] * 6 * mm,
             [1] * 3*mm + [3] * 2*mm + [2] * 3*mm + [3] * 2*mm]
             # [3] * 10]
        )
        # cell assignments
        true_z = torch.tensor([0] * cells_per_clone +
                              [1] * cells_per_clone +
                              [2] * cells_per_clone)

        cell_cn_profile = cn_profile[true_z, :]
        self.assertEqual(cell_cn_profile.shape, (cfg.n_cells, cfg.chain_length))

        obs = (cell_cn_profile * 100).T
        # introduce some randomness
        obs += torch.distributions.normal.Normal(0, 10).sample(obs.shape).int()

        self.assertEqual(obs.shape, (cfg.chain_length, cfg.n_cells))
        true_eps = torch.ones((cfg.n_nodes, cfg.n_nodes))
        true_eps[0, 1] = 1./(cfg.chain_length-1)
        true_eps[0, 2] = 3./(cfg.chain_length-1)

        fix_qc = qC(cfg, true_params={
            "c": cn_profile
        })

        fix_tree = nx.DiGraph()
        fix_tree.add_edges_from([(0, 1), (0, 2)], weight=.5)
        trees = [fix_tree] * cfg.wis_sample_size
        wis_weights = [1/cfg.wis_sample_size] * cfg.wis_sample_size

        for i in range(10):
            print(qeps.mean()[[0, 0], [1, 2]])
            qeps.update(trees, wis_weights, fix_qc.couple_filtering_probs)

        var_eps = qeps.mean()
        self.assertAlmostEqual(var_eps[0, 1], true_eps[0, 1], delta=.1)
        self.assertAlmostEqual(var_eps[0, 2], true_eps[0, 2], delta=.1)
        self.assertGreater(var_eps[0, 2], var_eps[0, 1])
