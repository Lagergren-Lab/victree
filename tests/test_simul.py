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
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qC, qTauUrn


class SimulTestCase(unittest.TestCase):

    def test_copy_tree_sim(self):
        torch.manual_seed(0)
        K = 5
        M = 2000
        A = 7
        eps_a = 1.
        eps_b = 10.
        eps_0 = 0.1
        tree = tests.utils_testing.get_tree_K_nodes_one_level(K)
        eps, c = simul.simulate_copy_tree_data(K, M, A, tree, eps_a, eps_b, eps_0)
        torch.equal(c[0, :], torch.ones((M,), dtype=torch.int) * 2)

        # For Large M, assert that the clone with most variance is gained from the edge with highest epsilon
        variances = torch.std(c.float(), dim=1)
        max_eps_arc = max(eps, key=eps.get)
        self.assertTrue(torch.argmax(variances) == max_eps_arc[1])

    def test_one_edge_tree(self):
        torch.manual_seed(0)
        logging.getLogger().setLevel("INFO")
        tree = tests.utils_testing.get_two_node_tree()
        n_nodes = len(tree.nodes)
        n_cells = 1000
        n_sites = 200
        n_copy_states = 7
        dir_delta = [1., 3.]
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        a0 = torch.tensor(10.0)
        b0 = torch.tensor(200.0)
        R_0 = 100.

        out = simul.simulate_data_total_GC_urn_model(tree, n_cells, n_sites, n_nodes, n_copy_states, R_0, eps_a=a0,
                                                     eps_b=b0, eps_0=1., alpha0=alpha0, beta0=beta0,
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

        torch.allclose(torch.mean(R.float()), torch.tensor(R_0))
        # think of interesting cases here

    def test_no_transitions_from_absorbing_state(self):
        utils.config.set_seed(0)
        logging.getLogger().setLevel("INFO")
        tree = tests.utils_testing.get_tree_three_nodes_chain()
        K = len(tree.nodes)
        N = 10
        M = 200
        A = 5
        dir_delta = [1., 3.]
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        a0 = torch.tensor(100.0)
        b0 = torch.tensor(200.0)
        config = Config(n_cells=N, chain_length=M, n_nodes=K, n_states=A)
        out = simul.simulate_full_dataset(config=config, eps_a=a0, eps_b=b0)
        y = out['obs']
        c = out['c']
        z = out['z']
        eps = out['eps']
        eps_0 = out['eps0']

        c1_0_idx = (torch.where(c[1] == 0)[0]).tolist()
        c2_0_idx = (torch.where(c[2] == 0)[0]).tolist()
        self.assertTrue(c1_0_idx in c2_0_idx)
