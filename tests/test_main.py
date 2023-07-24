import unittest

import numpy as np
import copy

import simul
from inference.victree import VICTree
from variational_distributions.joint_dists import VarTreeJointDist
from utils.config import set_seed, Config
from variational_distributions.var_dists import qCMultiChrom


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        set_seed(42)
        self.config = Config(n_nodes=4, n_states=5, n_cells=20, chain_length=200, step_size=0.1)
        self.obs = simul.simulate_full_dataset(config=self.config, dir_alpha=10.)["obs"]
        self.q = VarTreeJointDist(self.config, self.obs).initialize()
        self.copytree = VICTree(self.config, self.q, self.obs)

    def test_main(self):
        self.assertEqual(self.copytree.elbo, - np.infty)
        self.copytree.run(n_iter=5)
        mid_elbo = self.copytree.elbo
        self.copytree.run(n_iter=5)

        self.assertGreater(self.copytree.elbo, mid_elbo)

    def test_multichr_main(self):
        cfg = copy.deepcopy(self.config)
        # with hg19 referenced chr bins
        chr_df = simul.generate_chromosome_binning(cfg.chain_length)
        multi_dat = simul.simulate_full_dataset(cfg, chr_df=chr_df)

        obs = multi_dat['obs']

        q = VarTreeJointDist(cfg, obs, qc=qCMultiChrom(config=cfg)).initialize()
        victree = VICTree(cfg, q, obs)
        self.assertEqual(victree.elbo, - np.infty)
        victree.run(5)
        mid_elbo = victree.elbo
        victree.run(5)
        self.assertGreater(victree.elbo, mid_elbo)


if __name__ == '__main__':
    unittest.main()
