import unittest

import networkx as nx
import torch

from inference.copy_tree import VarDistFixedTree
from utils.config import set_seed, Config
from variational_distributions.var_dists import qC, qZ, qMuTau, qPi, qEpsilonMulti


class updatesTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # design simple test: fix all other variables
        # and update one var dist alone

        # set all seeds for reproducibility
        set_seed(101)

    def generate_test_dataset_fixed_tree(self) -> VarDistFixedTree:
        # obs with 15 cells, 5 each to different clone
        # in order, clone 0, 1, 2
        cells_per_clone = 10
        mm = 1  # change this to increase length
        chain_length = mm * 10  # total chain length shouldn't be more than 100, ow eps too small
        cfg = Config(n_nodes=3, n_states=5, n_cells=3 * cells_per_clone, chain_length=chain_length,
                     wis_sample_size=2, debug=True)
        # obs with 15 cells, 5 each to different clone
        # in order, clone 0, 1, 2
        true_cn_profile = torch.tensor(
            [[2] * 10*mm,
             [2] * 4*mm + [3] * 6*mm,
             [1] * 3*mm + [3] * 2*mm + [2] * 3*mm + [3] * 2*mm]
            # [3] * 10]
        )
        # cell assignments
        true_z = torch.tensor([0] * cells_per_clone +
                              [1] * cells_per_clone +
                              [2] * cells_per_clone)
        true_pi = torch.nn.functional.one_hot(true_z, num_classes=cfg.n_nodes).float()

        cell_cn_profile = true_cn_profile[true_z, :]
        self.assertEqual(cell_cn_profile.shape, (cfg.n_cells, cfg.chain_length))

        # mean and precision
        nu, lmbda = torch.tensor([1, 1])  # randomize mu for each cell with these hyperparameters
        true_mu = torch.randn(cfg.n_cells) / torch.sqrt(lmbda) + nu
        obs = (cell_cn_profile * true_mu[:, None]).T.clamp(min=0)
        self.assertEqual(obs.shape, (cfg.chain_length, cfg.n_cells))

        true_eps = torch.ones((cfg.n_nodes, cfg.n_nodes))
        true_eps[0, 1] = 1./(cfg.chain_length-1)
        true_eps[0, 2] = 3./(cfg.chain_length-1)

        # give true values to the other required dists
        fix_qc = qC(cfg, true_params={
            "c": true_cn_profile
        })

        fix_qz = qZ(cfg, true_params={
            "z": true_z
        })

        fix_qeps = qEpsilonMulti(cfg, true_params={
            "eps": true_eps
        })

        fix_qmt = qMuTau(cfg, true_params={
            "mu": true_mu,
            "tau": torch.ones(cfg.n_cells) * lmbda
        })

        fix_qpi = qPi(cfg, true_params={
            "pi": torch.ones(cfg.n_nodes) / 3.
        })

        fix_tree = nx.DiGraph()
        fix_tree.add_edges_from([(0, 1), (0, 2)], weight=.5)

        joint_q = VarDistFixedTree(cfg, fix_qc, fix_qz, fix_qeps,
                                   fix_qmt, fix_qpi, fix_tree, obs)
        return joint_q

    def test_update_qc(self):

        joint_q = self.generate_test_dataset_fixed_tree()
        cfg = joint_q.config
        obs = joint_q.obs
        fix_tree = joint_q.T
        fix_qeps = joint_q.eps
        fix_qz = joint_q.z
        fix_qmt = joint_q.mt

        trees = [fix_tree] * cfg.wis_sample_size
        wis_weights = [1/cfg.wis_sample_size] * cfg.wis_sample_size

        qc = qC(cfg)
        qc.initialize()

        for i in range(100):
            qc.update(obs, fix_qeps, fix_qz, fix_qmt,
                      trees=trees, tree_weights=wis_weights)

        self.assertTrue(torch.all(qc.couple_filtering_probs[0, :, 2, 2] > qc.couple_filtering_probs[0, :, 2, 0]))

        print(qc.single_filtering_probs)
        print(torch.argmax(qc.couple_filtering_probs.flatten(-2), dim=-1))

        self.assertEqual(qc.couple_filtering_probs[1, 3, 2, 3], qc.couple_filtering_probs[1, 3, 2, :].max())

        self.assertEqual(qc.couple_filtering_probs[2, 7, 2, 3], qc.couple_filtering_probs[2, 7, 1, :].max())


if __name__ == '__main__':
    unittest.main()
