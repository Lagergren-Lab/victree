import unittest

import networkx as nx
import torch
from pyro import poutine

import simul
import tests.utils_testing
from inference.copy_tree import VarDistFixedTree, CopyTree
from model.generative_model import GenerativeModel
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC


class VICtreeFixedTreeTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # build config
        self.config = Config(n_nodes=3, n_states=7, n_cells=20, chain_length=10)
        self.qc = qC(self.config)
        self.qc.initialize()
        self.qt = qT(self.config)
        self.qeps = qEpsilonMulti(self.config, 2, 5)  # skewed towards 0
        self.qz = qZ(self.config)
        self.qpi = qPi(self.config)
        self.qmt = qMuTau(self.config, loc=100, precision=.1,
                          shape=5, rate=5)

    def set_up_q(self, config):
        qc = qC(config)
        qt = qT(config)
        qeps = qEpsilonMulti(config)
        qz = qZ(config)
        qpi = qPi(config)
        qmt = qMuTau(config, loc=1, precision=.1, shape=5, rate=5)
        return qc, qt, qeps, qz, qpi, qmt

    def simul_data_pyro(self, data, n_cells, n_sites, n_copy_states, tree: nx.DiGraph):
        model_tree_markov = simul.model_tree_markov
        unconditioned_model = poutine.uncondition(model_tree_markov)
        C, y, z, pi, mu, sigma2, eps = unconditioned_model(data, n_cells, n_sites, n_copy_states, tree, )
        return C, y, z, pi, mu, sigma2, eps

    def test_small_tree(self):
        tree = tests.utils_testing.get_tree_three_nodes_balanced()
        n_cells = 20
        n_sites = 10
        n_copy_states = 7
        data = torch.ones((n_sites, n_cells))
        C, y, z, pi, mu, sigma2, eps = self.simul_data_pyro(data, n_cells, n_sites, n_copy_states, tree)
        print(f"C node 1 site 2: {C[1, 2]}")

        p = GenerativeModel(self.config, tree)
        q = VarDistFixedTree(self.config, self.qc, self.qz, self.qeps, self.qmt, self.qpi, tree, y)
        copy_tree = CopyTree(self.config, p, q, y)

        copy_tree.run(10)

    def test_large_tree(self):
        K = 10
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 20
        n_sites = 100
        n_copy_states = 7
        data = torch.ones((n_sites, n_cells))
        C, y, z, pi, mu, sigma2, eps = self.simul_data_pyro(data, n_cells, n_sites, n_copy_states, tree)
        print(f"C node 1 site 2: {C[1, 2]}")
        config = Config(n_nodes=K, chain_length=n_sites, n_cells=n_cells, n_states=n_copy_states)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        p = GenerativeModel(config, tree)
        q = VarDistFixedTree(config, qc, qz, qeps, qmt, qpi, tree, y)
        q.initialize()
        copy_tree = CopyTree(config, p, q, y)

        copy_tree.run(100)
        q_C = copy_tree.q.c.single_filtering_probs
        q_pi = copy_tree.q.pi.concentration_param
        print(f"True pi: {pi} \n q(pi): {q_pi}")