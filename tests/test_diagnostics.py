import unittest

import torch

from inference.copy_tree import CopyTree
from variational_distributions.joint_dists import VarTreeJointDist
from simul import generate_dataset_var_tree
from utils.config import set_seed, Config


class InitTestCase(unittest.TestCase):

    def setUp(self) -> None:
        set_seed(42)

    def test_halve_sieve(self):
        n_iter = 3
        for sieving_size, n_sieving_iter in [(2, 2), (3, 2), (2, 5), (5, 10)]:
            config = Config(n_nodes=3, n_cells=30, n_states=3, chain_length=5,
                            sieving_size=sieving_size, n_sieving_iter=n_sieving_iter, diagnostics=True)
            simul_joint = generate_dataset_var_tree(config)
            joint_q = VarTreeJointDist(config, simul_joint.obs).initialize()
            tot_num_iter = config.n_sieving_iter + n_iter + 1
            copytree = CopyTree(config, joint_q, joint_q.obs)
            joint_q.init_diagnostics(tot_num_iter)
            joint_q.update_diagnostics(0)
            copytree.halve_sieve()
            # assert that in any case, the diagnostics value in the last n_iter slots are all zero
            # and in the previous slots there's at least one value != 0
            self.assertTrue(torch.all(copytree.q.diagnostics_dict["C"][-n_iter:, ...] == 0),
                            msg=f"Not valid for config: {sieving_size}, {n_sieving_iter}")
            self.assertTrue(torch.any(copytree.q.diagnostics_dict["C"][-n_iter-1, ...] != 0),
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

        for q in copytree.q.get_units():
            for k in q.params_history.keys():
                self.assertEqual(len(q.params_history[k]), n_sieving_iter + n_iter + 1, msg=f"key issue: '{k}'")

