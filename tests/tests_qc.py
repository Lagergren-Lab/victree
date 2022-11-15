import unittest

import networkx as nx
import torch

from utils import tree_utils
from tests import utils_testing
from utils.config import Config
from variational_distributions.q_T import q_T
from variational_distributions.q_Z import qZ
from variational_distributions.q_epsilon import qEpsilon
from variational_distributions.variational_hmm import CopyNumberHmm
from variational_distributions.variational_normal import qMuTau


class qCTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # build config
        self.config = Config()
        self.qc = CopyNumberHmm(self.config)
        self.qt = q_T(self.config)
        self.qeps = qEpsilon(self.config, 2, 5) # skewed towards 0
        self.qz = qZ(self.config)
        self.qmt = qMuTau(self.config, loc = 100, precision = .1,
                shape = 5, rate = 5)
        self.obs = torch.randint(low = 50, high = 150,
                size = (self.config.chain_length, self.config.n_cells))
                

    def test_update(self):
        self.qc.update(self.obs, self.qt, self.qeps, self.qz, self.qmt)

    def test_expectation_size(self):
        exp_alpha1, exp_alpha2 = self.qc.exp_alpha(self.qeps)
        self.assertEqual(exp_alpha1.shape, (self.config.n_nodes, self.config.chain_length, self.config.n_states))
        self.assertEqual(exp_alpha2.shape, (self.config.n_nodes, self.config.chain_length, self.config.n_states, self.config.n_states))

        tree = nx.random_tree(self.config.n_nodes, create_using = nx.DiGraph)
        exp_eta1, exp_eta2 = self.qc.exp_eta(self.obs, tree, self.qeps, self.qz, self.qmt)
        self.assertEqual(exp_eta1.shape, (self.config.n_nodes, self.config.chain_length, self.config.n_states))
        self.assertEqual(exp_eta2.shape, (self.config.n_nodes, self.config.chain_length, self.config.n_states, self.config.n_states))

        






