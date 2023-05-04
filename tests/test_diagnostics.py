import os.path
import unittest

import numpy as np
import torch

from inference.copy_tree import CopyTree
from utils.data_handling import write_checkpoint_h5
from variational_distributions.joint_dists import VarTreeJointDist
from simul import generate_dataset_var_tree
from utils.config import set_seed, Config
from variational_distributions.var_dists import qT


class InitTestCase(unittest.TestCase):

    def setUp(self) -> None:
        set_seed(42)
        self.output_dir = "./test_output"
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def test_halve_sieve(self):
        n_iter = 3
        for sieving_size, n_sieving_iter in [(2, 2), (3, 2), (2, 5), (5, 10)]:
            config = Config(n_nodes=3, n_cells=30, n_states=3, chain_length=5,
                            sieving_size=sieving_size, n_sieving_iter=n_sieving_iter, diagnostics=True)
            simul_joint = generate_dataset_var_tree(config)
            joint_q = VarTreeJointDist(config, simul_joint.obs).initialize()
            tot_num_iter = config.n_sieving_iter + n_iter + 1
            copytree = CopyTree(config, joint_q, joint_q.obs)
            copytree.halve_sieve()
            # assert that in any case, the diagnostics value in the last n_iter slots are all zero
            # and in the previous slots there's at least one value != 0
            self.assertEqual(len(copytree.q.c.params_history["single_filtering_probs"]), config.n_sieving_iter + 1,
                             msg=f"Not valid for config: {sieving_size}, {n_sieving_iter}")

    def test_progress_tracking(self):
        n_iter = 3
        n_sieving_iter = 3
        config = Config(n_nodes=3, n_cells=30, n_states=3, chain_length=5,
                        sieving_size=3, n_sieving_iter=n_sieving_iter, diagnostics=True)
        simul_joint = generate_dataset_var_tree(config)
        joint_q = VarTreeJointDist(config, simul_joint.obs).initialize()
        copytree = CopyTree(config, joint_q, joint_q.obs)
        copytree.run(n_iter)

        for q in copytree.q.get_units() + [copytree.q]:
            for k in q.params_history.keys():
                self.assertEqual(len(q.params_history[k]), n_sieving_iter + n_iter + 1, msg=f"key issue: '{k}'")
                self.assertTrue(isinstance(q.params_history[k][-1], np.ndarray),
                                msg=f"param {k} is of type {type(q.params_history[k][-1])} but it should be np.ndarray")

        write_checkpoint_h5(copytree, path=os.path.join(self.output_dir, "checkpoint_" + str(copytree) + ".h5"))
