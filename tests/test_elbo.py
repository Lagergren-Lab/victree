import logging
import unittest
from utils.config import Config

from variational_distributions.variational_distribution import VariationalDistribution
from variational_distributions.q_Z import qZ
from model.generative_model import GenerativeModel
from inference.copy_tree import CopyTree

class TestElbo(unittest.TestCase):

    def test_elbo_decrease_exception(self):
        
        config = Config()
        p = GenerativeModel()
        q = VariationalDistribution(config)
        q_z = qZ(config)
        copy_tree = CopyTree(p, q, q, q_z)

        copy_tree.elbo = -100. 
        # compute_elbo currently outputs -1000
        # elbo is not updated until step() is not fully implemented

        with self.assertRaises(ValueError) as ve_info:
            logging.info(str(ve_info))
            copy_tree.run(10)

