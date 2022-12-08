import unittest

import torch

from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qEpsilon, qZ, qMuTau, qC

class qmtTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # build config
        self.config = Config(chain_length=15, n_nodes=3, n_cells=5, n_states=7)
        self.qc = qC(self.config)
        self.qc.initialize()
        self.qt = qT(self.config)
        self.qeps = qEpsilonMulti(self.config, 2, 5) # skewed towards 0
        self.qz = qZ(self.config)
        self.qz.initialize()
        self.qmt = qMuTau(self.config, loc = 100, precision = .1,
                shape = 5, rate = 5)
        self.obs = torch.randint(low = 50, high = 150,
                size = (self.config.chain_length, self.config.n_cells), dtype=torch.float)
        self.sum_M_y2 = torch.sum(self.obs**2)

    def test_update(self):
        mu, lmbda, alpha, beta = self.qmt.update(qc=self.qc, qz=self.qz, obs=self.obs, sum_M_y2=self.sum_M_y2)

    def test_expectation_size(self):
        exp_log_emission = self.qmt.exp_log_emission(self.obs)
        self.assertEqual(exp_log_emission.shape, (self.config.n_cells,
                                                  self.config.chain_length,
                                                  self.config.n_states))

