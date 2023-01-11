import logging
import unittest
import os
from pathlib import Path

from utils.config import Config
from utils.data_handling import read_sc_data

from variational_distributions.var_dists import qZ, qC, qMuTau, qEpsilon, qPi, qT
from model.generative_model import GenerativeModel
from inference.copy_tree import CopyTree, JointVarDist

class TestElbo(unittest.TestCase):

    def setUp(self) -> None:
        self.proj_dir = Path(__file__).parent.parent
        return super().setUp()

    def test_elbo_decrease_exception(self):
        #TODO: Obsolete test for sampling estimated ELBO
        config = Config()
        p = GenerativeModel(config)
        q_c = qC(config)
        q_z = qZ(config)
        q_t = qT(config)
        q_pi = qPi(config)
        q_eps = qEpsilon(config, 1., 1.)
        q_mutau = qMuTau(config, loc=100., precision_factor=.1, shape=5., rate=5.)
        _, _, obs = read_sc_data(self.proj_dir / 'obs_example.txt')
        obs = obs.float()
        joint_dist = JointVarDist(config, 
                                  q_c, q_z, q_t, q_eps, q_mutau, q_pi, obs)

        copy_tree = CopyTree(config, p, joint_dist, obs)

        copy_tree.elbo = -100. 
        # compute_elbo currently outputs -1000
        # elbo is not updated until step() is not fully implemented

        with self.assertRaises(ValueError) as ve_info:
            logging.info(str(ve_info))
            copy_tree.run(5)

