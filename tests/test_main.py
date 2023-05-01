import unittest

import numpy as np

import simul
from inference.copy_tree import CopyTree
from variational_distributions.joint_dists import VarTreeJointDist
from utils.config import set_seed, Config


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        set_seed(42)
        self.config = Config(n_nodes=4, n_states=5, n_cells=20, chain_length=200, step_size=0.1)
        self.obs = simul.simulate_full_dataset(config=self.config, dir_alpha=10.)["obs"]
        self.q = VarTreeJointDist(self.config, self.obs).initialize()
        self.copytree = CopyTree(self.config, self.q, self.obs)

    def test_main(self):
        self.assertEqual(self.copytree.elbo, - np.infty)
        self.copytree.run(5)
        mid_elbo = self.copytree.elbo
        self.copytree.run(5)

        self.assertGreater(self.copytree.elbo, mid_elbo)




if __name__ == '__main__':
    unittest.main()
