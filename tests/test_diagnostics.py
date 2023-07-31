import os.path
import unittest

import h5py
import numpy as np

from inference.victree import VICTree
from utils.data_handling import write_checkpoint_h5, load_h5_pseudoanndata, read_last_it_from_checkpoint
from variational_distributions.joint_dists import VarTreeJointDist
from simul import generate_dataset_var_tree
from utils.config import set_seed, Config
from variational_distributions.var_dists import qCMultiChrom


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
            copytree = VICTree(config, joint_q, joint_q.obs)
            copytree.halve_sieve()
            # assert that in any case, the diagnostics value in the last n_iter slots are all zero
            # and in the previous slots there's at least one value != 0
            self.assertEqual(len(copytree.q.c.params_history["single_filtering_probs"]), config.n_sieving_iter + 1,
                             msg=f"Not valid for config: {sieving_size}, {n_sieving_iter}")

    def test_progress_tracking(self):
        n_iter = 30
        n_sieving_iter = 3
        config = Config(n_nodes=3, n_cells=30, n_states=3, chain_length=5, n_run_iter=n_iter,
                        sieving_size=3, n_sieving_iter=n_sieving_iter, diagnostics=True, out_dir=self.output_dir)
        # if check below does not hold, first checkpoint save would happen at the end of inference
        # and params_history would not be reset (which is here subject to test)
        self.assertGreater(config.n_run_iter, config.save_progress_every_niter)

        simul_joint = generate_dataset_var_tree(config)
        joint_q = VarTreeJointDist(config, simul_joint.obs, qt=simul_joint.t).initialize()
        victree = VICTree(config, joint_q, joint_q.obs)

        # new checkpoint file
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_" + str(victree) + ".h5")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        victree.halve_sieve()
        for i in range(n_iter):
            victree.step()

        for q in victree.q.get_units() + [victree.q]:
            for k in q.params_history.keys():
                self.assertEqual(len(q.params_history[k]), config.n_run_iter % config.save_progress_every_niter,
                                 msg=f"key issue: '{k}'")
                self.assertTrue(isinstance(q.params_history[k][-1], np.ndarray),
                                msg=f"param {k} is of type {type(q.params_history[k][-1])} but it should be np.ndarray")

        write_checkpoint_h5(victree, path=checkpoint_path)

    def test_append_checkpoint(self):

        file_path = os.path.join(self.output_dir, "append_test.h5")
        if os.path.exists(file_path):
            os.remove(file_path)
        with h5py.File(file_path, 'a') as f:
            self.assertEqual(len(f.keys()), 0)
            g = f.create_group("group")
            dim2 = 5
            init_data_size = 5
            dset = g.create_dataset("data", data=np.arange(init_data_size * dim2).reshape((init_data_size, -1)),
                                    maxshape=(None, dim2),
                                    chunks=True)
            # keys only view first level (not recursive)
            self.assertEqual(len(f.keys()), 1)

            new_data_size = 4
            dset.resize(len(dset) + new_data_size, axis=0)
            dset[-new_data_size:] = np.arange(new_data_size * dim2).reshape((new_data_size, -1))
            self.assertEqual(dset.shape, (init_data_size + new_data_size, dim2))

    def test_multichr_history_length(self):
        n_iter = 30
        n_sieving_iter = 3

        config = Config(n_nodes=3, n_cells=30, n_states=3, chain_length=20, n_run_iter=n_iter,
                        sieving_size=3, n_sieving_iter=n_sieving_iter, diagnostics=True, out_dir=self.output_dir)
        # if check below does not hold, first checkpoint save would happen at the end of inference
        # and params_history would not be reset (which is here subject to test)
        self.assertGreater(config.n_run_iter, config.save_progress_every_niter)

        simul_joint = generate_dataset_var_tree(config, chrom=3)
        qc_multi = qCMultiChrom(config)
        joint_q = VarTreeJointDist(config, simul_joint.obs, qc=qc_multi).initialize()
        victree = VICTree(config, joint_q, joint_q.obs)

        # new checkpoint file
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_" + str(victree) + ".h5")
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        victree.halve_sieve()
        for i in range(n_iter):
            victree.step()

        for q in victree.q.get_units() + [victree.q]:
            for k in q.params_history.keys():
                self.assertEqual(len(q.params_history[k]), config.n_run_iter % config.save_progress_every_niter,
                                 msg=f"key issue: '{k}'")
                self.assertTrue(isinstance(q.params_history[k][-1], np.ndarray),
                                msg=f"param {k} is of type {type(q.params_history[k][-1])} but it should be np.ndarray")

        write_checkpoint_h5(victree, path=checkpoint_path)

        # check what is saved in the checkpoint
        h5_checkpoint = load_h5_pseudoanndata(checkpoint_path)
        for dist_name in h5_checkpoint.keys():
            for param_name in h5_checkpoint[dist_name].keys():
                ds = h5_checkpoint[dist_name][param_name]
                self.assertEqual(ds.shape[0], config.n_sieving_iter + config.n_run_iter + 1)


    def test_load_checkpoint(self):
        test_checkpoint_file_path = os.path.join(self.output_dir, "checkpoint_k3a3n30m5.h5")
        if not os.path.exists(test_checkpoint_file_path):
            raise Exception("Test checkpoint file doesn't exist! Run test 'test_progress_tracking' first.")

        loaded_checkpoint = read_last_it_from_checkpoint(test_checkpoint_file_path)

        self.assertTrue(loaded_checkpoint['qT']['weight_matrix'].shape == (14, 3, 3))



