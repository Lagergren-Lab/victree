import logging
import unittest

import torch
from utils.config import Config
from variational_distributions.q_T import q_T
from variational_distributions.q_epsilon import qEpsilon

from variational_distributions.variational_distribution import VariationalDistribution
from variational_distributions.q_Z import qZ
from model.generative_model import GenerativeModel
from inference.copy_tree import CopyTree, JointVarDist
from variational_distributions.variational_hmm import CopyNumberHmm
from variational_distributions.variational_normal import qMuTau

class TestElbo(unittest.TestCase):

    def test_elbo_decrease_exception(self):
        
        config = Config()
        p = GenerativeModel()
        q_c = CopyNumberHmm(config)
        q_z = qZ(config)
        q_t = q_T(config)
        q_eps = qEpsilon(config, 1, 1)
        q_mutau = qMuTau(config, loc = 100, precision = .1,
                shape = 5, rate = 5)
        obs = torch.ones((config.n_states, config.n_cells))
        joint_dist = JointVarDist(config, q_c, q_z, q_t, q_eps, q_mutau, obs)

        copy_tree = CopyTree(config, p, joint_dist, obs)

        copy_tree.elbo = -100. 
        # compute_elbo currently outputs -1000
        # elbo is not updated until step() is not fully implemented

        with self.assertRaises(ValueError) as ve_info:
            logging.info(str(ve_info))
            copy_tree.run(10)

