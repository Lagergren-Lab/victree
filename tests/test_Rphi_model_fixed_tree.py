import logging
import random
import unittest

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as f

import simul
import tests.utils_testing
import utils.config
from inference.victree import VICTree
from variational_distributions.joint_dists import FixedTreeJointDist
from tests import model_variational_comparisons
from tests.utils_testing import simul_data_pyro_full_model, simulate_full_dataset_no_pyro
from utils import visualization_utils
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qC, qTauUrn, qPhi


class RPhiModelFixedTreeTestCase(unittest.TestCase):

    def set_up_q(self, config, R, gc):
        qc = qC(config)
        qt = qT(config)
        qeps = qEpsilonMulti(config)
        qz = qZ(config)
        qpi = qPi(config)
        return qc, qt, qeps, qz, qpi

    def test_one_edge_tree_poisson(self):
        torch.manual_seed(0)
        logging.getLogger().setLevel("INFO")
        tree = tests.utils_testing.get_two_node_tree()
        n_nodes = len(tree.nodes)
        n_cells = 1000
        n_sites = 200
        n_copy_states = 7
        dir_delta = torch.tensor([1., 3.])
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        a0 = torch.tensor(10.0)
        b0 = torch.tensor(200.0)
        eps_0 = 1.
        R_0 = 1000

        out = simul.simulate_data_total_GC_urn_model(tree, n_cells, n_sites, n_nodes, n_copy_states, R_0,
                                                     emission_model='poisson',
                                                     eps_a=a0, eps_b=b0, eps_0=eps_0, alpha0=alpha0, beta0=beta0,
                                                     dir_delta=dir_delta)
        x = out['x']
        R = out['R']
        gc = out['gc']
        phi = out['phi']
        c = out['c']
        z = out['z']
        pi = out['pi']
        eps = out['eps']
        eps_0 = out['eps0']
        print(f"Epsilon: {eps}")

        config = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3,
                        debug=False, diagnostics=False)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)

        qc, qt, qeps, qz, qpi = self.set_up_q(config, R, gc)
        phi_init = 1.
        qpsi = qPhi(config, phi_init, x, gc, R, n_copy_states, emission_model="poisson")
        q = FixedTreeJointDist(config, qc, qz, qeps, qpsi, qpi, tree, x)
        q.initialize()
        copy_tree = VICTree(config, q, x)

        # Act
        copy_tree.run(n_iter=50)

        # Assert
        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_urn_model_comparisons(x, R, gc, phi, c, z, pi, eps, qc, qz, qpi, qpsi, qeps)

    def test_three_node_tree(self):
        torch.manual_seed(0)
        tree = tests.utils_testing.get_tree_three_nodes_balanced()
        n_nodes = len(tree.nodes)
        n_cells = 1000
        n_sites = 200
        n_copy_states = 7
        dir_delta = torch.tensor([3., 10., 10.])
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        a0 = torch.tensor(5.0)
        b0 = torch.tensor(200.0)
        eps_0 = 1.
        R_0 = n_sites * 10
        out = simul.simulate_data_total_GC_urn_model(tree, n_cells, n_sites, n_nodes, n_copy_states, R_0, eps_a=a0,
                                                     eps_b=b0, eps_0=eps_0, alpha0=alpha0, beta0=beta0,
                                                     dir_delta=dir_delta)
        x = out['x']
        R = out['R']
        gc = out['gc']
        phi = out['phi']
        c = out['c']
        z = out['z']
        pi = out['pi']
        tau = out['psi']
        eps = out['eps']
        eps_0 = out['eps0']

        print(f"Epsilon: {eps}")
        print(f"pi: {pi}")
        config = Config(step_size=0.3, n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites,
                        debug=False, diagnostics=False)
        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)

        qc, qt, qeps, qz, qpi = self.set_up_q(config, R, gc)
        phi_init = 1.
        qpsi = qPhi(config, phi_init, x, gc, R, n_copy_states, emission_model="poisson")
        q = FixedTreeJointDist(config, qc, qz, qeps, qpsi, qpi, tree, x)
        q.initialize()
        copy_tree = VICTree(config, q, x)

        copy_tree.run(n_iter=50)

        # Assert
        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_urn_model_comparisons(x, R, gc, phi, c, z, pi, eps, qc, qz, qpi, qpsi, qeps)


    def test_large_tree_poisson(self):
        torch.manual_seed(0)
        n_nodes = 7
        tree = tests.utils_testing.get_tree_K_nodes_random(n_nodes)
        n_cells = 1000
        n_sites = 200
        n_copy_states = 7
        dir_delta = torch.ones(n_nodes,) * 5.
        dir_delta[0] = 2.
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        a0 = torch.tensor(5.0)
        b0 = torch.tensor(200.0)
        eps_0 = 1.
        R_0 = n_sites * 10
        out = simul.simulate_data_total_GC_urn_model(tree, n_cells, n_sites, n_nodes, n_copy_states, R_0, eps_a=a0,
                                                     eps_b=b0, eps_0=eps_0, alpha0=alpha0, beta0=beta0,
                                                     dir_delta=dir_delta)
        x = out['x']
        R = out['R']
        gc = out['gc']
        phi = out['phi']
        c = out['c']
        z = out['z']
        pi = out['pi']
        tau = out['psi']
        eps = out['eps']
        eps_0 = out['eps0']

        print(f"Epsilon: {eps}")
        print(f"pi: {pi}")
        config = Config(step_size=0.3, n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites,
                        debug=False, diagnostics=False)
        qc, qt, qeps, qz, qpi = self.set_up_q(config, R, gc)
        phi_init = 1.
        qpsi = qPhi(config, phi_init, x, gc, R, n_copy_states, emission_model="poisson")
        q = FixedTreeJointDist(config, qc, qz, qeps, qpsi, qpi, tree, x)
        q.initialize()
        q.z.initialize(z_init='kmeans', obs=x)

        print(f"-------------- qZ after init -------------------")
        model_variational_comparisons.compare_qZ_and_true_Z(z, q.z)
        print(f"-------------- init complete -------------------")

        copy_tree = VICTree(config, q, x)

        copy_tree.run(n_iter=50)

        # Assert
        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_urn_model_comparisons(x, R, gc, phi, c, z, pi, eps, qc, qz, qpi, qpsi, qeps)