import unittest

import networkx as nx
import torch

from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qEpsilon, qZ, qMuTau, qC


class qCTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # build config
        self.config = Config()
        self.qc = qC(self.config)
        self.qc.initialize()
        self.qt = qT(self.config)
        self.qeps = qEpsilonMulti(self.config, 2, 5) # skewed towards 0
        self.qz = qZ(self.config)
        self.qmt = qMuTau(self.config, loc = 100, precision = .1,
                shape = 5, rate = 5)
        self.obs = torch.randint(low = 50, high = 150,
                size = (self.config.chain_length, self.config.n_cells))
                

    def test_update(self):
        self.qc.update(self.obs, self.qt, self.qeps, self.qz, self.qmt)

    def test_expectation_size(self):
        tree = nx.random_tree(self.config.n_nodes, create_using = nx.DiGraph)
        exp_alpha1, exp_alpha2 = self.qc.exp_alpha(tree, self.qeps)
        self.assertEqual(exp_alpha1.shape, (self.config.n_nodes, self.config.chain_length, self.config.n_states))
        self.assertEqual(exp_alpha2.shape, (self.config.n_nodes, self.config.chain_length, self.config.n_states, self.config.n_states))

        exp_eta1, exp_eta2 = self.qc.exp_eta(self.obs, tree, self.qeps, self.qz, self.qmt)
        self.assertEqual(exp_eta1.shape, (self.config.n_nodes, self.config.chain_length, self.config.n_states))
        self.assertEqual(exp_eta2.shape, (self.config.n_nodes, self.config.chain_length, self.config.n_states, self.config.n_states))

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







