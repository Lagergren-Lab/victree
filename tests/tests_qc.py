import unittest

import networkx as nx
import torch

from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qEpsilon, qZ, qMuTau, qC
from inference.copy_tree import JointVarDist


class qCTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # build config
        self.config = Config()
        self.qc = qC(self.config)
        self.qc.initialize()
        self.qt = qT(self.config)
        self.qeps = qEpsilonMulti(self.config, 2, 5)  # skewed towards 0
        self.qz = qZ(self.config)
        self.qmt = qMuTau(self.config, loc=100, precision=.1,
                          shape=5, rate=5)
        self.obs = torch.randint(low=50, high=150,
                                 size=(self.config.chain_length, self.config.n_cells))

    def test_update(self):
        # design simple test: fix all other variables
        # and update the q_c params
        cfg = Config(n_nodes=3, n_states=5, n_cells=15, chain_length=10,
                     wis_sample_size=2)
        # obs with 15 cells, 5 each to different clone
        obs = torch.tensor([
            [[200] * 10] * 5 +
            [[200] * 4 + [300] * 6] * 5 +
            [[100] * 8 + [200] * 2] * 5
        ]).T
        self.assertEqual(obs.shape, (cfg.chain_length, cfg.n_cells))

        joint_var = JointVarDist(cfg)
        self.qc.update(self.obs, self.qt, self.qeps, self.qz, self.qmt)

    def test_expectation_size(self):
        tree = nx.random_tree(self.config.n_nodes, create_using=nx.DiGraph)
        exp_alpha1, exp_alpha2 = self.qc.exp_alpha(tree, self.qeps)
        self.assertEqual(exp_alpha1.shape, (self.config.n_nodes, self.config.chain_length, self.config.n_states))
        self.assertEqual(exp_alpha2.shape,
                         (self.config.n_nodes, self.config.chain_length-1, self.config.n_states, self.config.n_states))

        exp_eta1, exp_eta2 = self.qc.exp_eta(self.obs, tree, self.qeps, self.qz, self.qmt)
        self.assertEqual(exp_eta1.shape, (self.config.n_nodes, self.config.chain_length, self.config.n_states))
        self.assertEqual(exp_eta2.shape,
                         (self.config.n_nodes, self.config.chain_length-1, self.config.n_states, self.config.n_states))

    def test_ELBO(self):
        L_list = [1, 2, 5, 10, 20]
        trees = []
        weights = []
        L_prev = 0
        for L in L_list:
            new_trees, new_weights = self.qt.get_trees_sample(L=(L - L_prev))
            trees = trees + new_trees
            weights = weights + new_weights
            res_1 = self.qc.elbo(trees, weights, self.qeps)
            print(f" {res_1}")
            L_prev = L

    def test_entropy_higher_for_random_transitions_than_deterministic_transitions(self):
        K = 3
        M = 5
        N = 2
        A = 7
        config_1 = Config(n_nodes=K, n_cells=N, chain_length=M, n_states=A)
        qc_1 = qC(config_1)
        qc_1.initialize()
        entropy_rand = qc_1.entropy()
        print(f"Random: {entropy_rand}")
        qc_2 = qC(config_1)
        for k in range(K):
            for m in range(M-1):
                qc_2.eta2[k, m] = torch.diag(torch.ones(A, ))
        entropy_rand_deterministic = qc_2.entropy()
        print(f"Deterministic: {entropy_rand_deterministic}")
        self.assertGreater(entropy_rand, entropy_rand_deterministic)

    def test_entropy_lower_for_random_transitions_than_uniform_transitions(self):
        rand_res = self.qc.entropy()
        qc_1 = qC(self.config)
        qc_1.uniform_init()
        uniform_res = qc_1.entropy()
        print(f"Entropy random eta_2: {rand_res} - uniform eta_2: {uniform_res}")
        self.assertLess(rand_res, uniform_res)
