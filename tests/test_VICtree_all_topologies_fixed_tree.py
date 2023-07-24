import logging
import unittest

import torch

import tests.utils_testing
from inference.victree import VICTree
from variational_distributions.joint_dists import FixedTreeJointDist
from tests.utils_testing import simulate_full_dataset_no_pyro
from utils import tree_utils
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent


class VICtreeFixedTreeTestCase(unittest.TestCase):

    def set_up_q(self, config):
        qc = qC(config)
        qt = qT(config)
        qeps = qEpsilonMulti(config)
        qz = qZ(config)
        qpi = qPi(config)
        qmt = qMuTau(config)
        return qc, qt, qeps, qz, qpi, qmt

    @unittest.skip('not implemented yet')
    def test_ELBO_of_all_topologies(self):
        torch.manual_seed(0)
        logging.getLogger().setLevel("INFO")
        K = 4
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_nodes = len(tree.nodes)
        n_cells = 100
        n_sites = 50
        n_copy_states = 7
        dir_alpha = [1., 3.]
        nu_0 = 10.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 5.0
        b0 = 200.0

        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha)
        config = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=1.0,
                        debug=False, diagnostics=True)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)

        T_list = tree_utils.get_all_topologies(K)
        n_top = len(T_list)
        elbos = torch.zeros(n_top,)
        for i, T in enumerate(T_list):
            qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
            q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, T, y)
            q.initialize()
            copy_tree = VICTree(config, q, y)

            # Act
            copy_tree.run(n_iter=10)
            elbos[i] = copy_tree.elbo

        # TODO: Plot elbos vs distance from true tree


