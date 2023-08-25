import logging
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    @unittest.skip('long exec test')
    def test_ELBO_of_all_topologies(self):
        torch.manual_seed(0)
        logging.getLogger().setLevel("INFO")
        K = 4
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_nodes = len(tree.nodes)
        n_cells = 100
        n_sites = 500
        n_copy_states = 7
        dir_alpha = [1., 3., 3., 3.]
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 500.0

        y, c, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha)
        config = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=1.0,
                        debug=False, diagnostics=False, split=True)
        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self.id().replace(".", "/"),
                                                                       base_dir='../test_output')

        T_list = tree_utils.get_all_tree_topologies(K)
        n_top = len(T_list)
        elbos = torch.zeros(n_top,)
        for i, T in enumerate(T_list):
            qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
            qeps = qEpsilonMulti(config, alpha_prior=a0, beta_prior=b0)
            q = FixedTreeJointDist(y, config, qc, qz, qeps, qmt, qpi, T)
            q.initialize()
            victree = VICTree(config, q, y)

            # Act
            victree.run(n_iter=10)
            elbos[i] = victree.elbo

        # labeled distances
        distances = tree_utils.distances_to_true_tree(tree, T_list, labeled_distance=True)
        plt.plot(distances, elbos, 'o')
        plt.savefig(test_dir_name + '/labeled_distances_plot')
        plt.close()

        T_max_distance_idx = np.argmax(distances)
        T_min_distance_idx = np.argmin(distances)
        T_max_distance = T_list[T_max_distance_idx]
        T_min_distance = T_list[T_min_distance_idx]

        out_df = pd.DataFrame([[tree.edges,
                                 T_max_distance.edges, distances[T_max_distance_idx], elbos[T_max_distance_idx],
                                 T_min_distance.edges, distances[T_min_distance_idx], elbos[T_min_distance_idx]]],
                               columns=['true_tree',
                                        'T_max_distance', 'distance max', 'elbo max',
                                        'T_min_distance', 'distance min', 'elbo min'])
        out_df.to_csv(test_dir_name + '/edges_max_min_labeled_distance_trees.csv')

        # unlabeled distances
        distances = tree_utils.distances_to_true_tree(tree, T_list, labeled_distance=False)
        plt.plot(distances, elbos, 'o')
        plt.savefig(test_dir_name + '/labeled_distances_plot')
        plt.close()

        T_max_distance_idx = np.argmax(distances)
        T_min_distance_idx = np.argmin(distances)
        T_max_distance = T_list[T_max_distance_idx]
        T_min_distance = T_list[T_min_distance_idx]

        out_df = pd.DataFrame([[tree.edges,
                                T_max_distance.edges, distances[T_max_distance_idx], elbos[T_max_distance_idx],
                                T_min_distance.edges, distances[T_min_distance_idx], elbos[T_min_distance_idx]]],
                              columns=['true_tree',
                                       'T_max_distance', 'distance max', 'elbo max',
                                       'T_min_distance', 'distance min', 'elbo min'])
        out_df.to_csv(test_dir_name + '/edges_max_min_unlabeled_distance_trees.csv')

        # Min max ELBO trees
        T_max_elbo_idx = torch.argmax(elbos)
        T_min_elbo_idx = torch.argmin(elbos)
        T_max_elbo = T_list[T_max_elbo_idx]
        T_min_elbo = T_list[T_min_elbo_idx]

        out_df = pd.DataFrame([[tree.edges,
                                T_max_elbo.edges, distances[T_max_elbo_idx], elbos[T_max_elbo_idx],
                                T_min_elbo.edges, distances[T_min_elbo_idx], elbos[T_min_elbo_idx]]],
                              columns=['true_tree',
                                       'T_max_elbo', 'distance max', 'elbo max',
                                       'T_min_elbo', 'distance min', 'elbo min'])
        out_df.to_csv(test_dir_name + '/edges_max_min_elbo_trees.csv')

