import unittest

import networkx as nx
import torch

from utils.config import Config
from variational_distributions.q_T import q_T
from variational_distributions.q_Z import qZ
from variational_distributions.q_epsilon import qEpsilon
from variational_distributions.variational_hmm import CopyNumberHmm
from variational_distributions.variational_normal import qMuTau

class qmtTestCase(unittest.TestCase):

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
        pass

    def test_expectation_size(self):
        exp_log_emission = self.qmt.exp_log_emission(self.obs)
        self.assertEqual(exp_log_emission.shape, (self.config.n_cells,
                                                  self.config.chain_length,
                                                  self.config.n_states))

