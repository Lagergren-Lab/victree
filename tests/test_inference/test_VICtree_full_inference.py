import logging
import os.path
import random
import unittest

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as f
import numpy as np

import simul
import tests.utils_testing
import utils.config
from inference.victree import VICTree
from variational_distributions.joint_dists import FixedTreeJointDist, VarTreeJointDist
from tests import model_variational_comparisons
from tests.utils_testing import simulate_full_dataset_no_pyro
from utils import visualization_utils, data_handling, tree_utils
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent

@unittest.skip("Too long")
class VICtreeFullInferenceTestCase(unittest.TestCase):

    def setUp(self) -> None:
        utils.config.set_seed(0)

    def set_up_q(self, config):
        qc = qC(config)
        qt = qT(config)
        qeps = qEpsilonMulti(config)
        qz = qZ(config)
        qpi = qPi(config, delta_prior=.8 * config.n_cells / config.n_nodes)
        qmt = qMuTau(config,
                     nu_prior=1., lambda_prior=config.chain_length * 2,
                     alpha_prior=1., beta_prior=1.)
        return qc, qt, qeps, qz, qpi, qmt

    def test_K_node_tree(self):
        torch.manual_seed(0)
        K = 7
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 100
        n_sites = 200
        n_copy_states = 7
        delta = np.ones(K) * 3.
        delta[0] = 1.
        delta = list(delta)
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 200.0
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=delta
                                                                        )

        config = Config(n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3,
                        diagnostics=False, qT_temp=200., split='ELBO')

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = VarTreeJointDist(config, y, qc, qz, qt, qeps, qmt, qpi)
        q.initialize()
        copy_tree = VICTree(config, q, y, draft=True)

        copy_tree.run(n_iter=100)

        # Assert
        torch.set_printoptions(precision=2)
        out = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps,
                                                                q_c=copy_tree.q.c,
                                                                q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                                                                q_mt=copy_tree.q.mt, q_eps=qeps)
        ari, perm, acc = (out['ari'], out['perm'], out['acc'])
        # Inspect trees
        sample_size = 30
        qt.g_temp = 5. * qt.temp
        T_list, w_T = qt.get_trees_sample(sample_size=sample_size)
        T_map = nx.maximum_spanning_arborescence(qt.weighted_graph)
        T_map_relabeled = tree_utils.relabel_trees([T_map], perm)[0]
        relabeled_trees = tree_utils.relabel_trees(T_list, labeling=perm)
        unique_seq, unique_seq_idx, multiplicity = tree_utils.unique_trees_and_multiplicity(relabeled_trees)
        print(f"True tree edges: {tree.edges}")
        print(f"MAP tree edges: {T_map_relabeled.edges}")
        print(f"Sampled tree edges: {[T.edges for T in relabeled_trees]} weights: {w_T}")
        print(f"Sample n unique: {len(unique_seq)} of {sample_size} with multiplicities: "
              f"{[m for m in multiplicity if m >= 2]} and "
              f"{int(np.sum([m for m in multiplicity if m == 1]))} trees with multiplicity = 1")

        print(f"N True tree in sampled trees: {np.sum([tree.edges == T.edges for T in relabeled_trees])}")
        print(f"N T_map in sampled trees: {np.sum([tree.edges == T_map_relabeled.edges for T in relabeled_trees])}")

        true_q = FixedTreeJointDist(
            y, config,
            qC(config, true_params={'c': C}),
            qZ(config, true_params={'z': z}),
            qEpsilonMulti(config, true_params={'eps': eps}),
            qMuTau(config, true_params={'mu': mu, 'tau': tau}),
            qPi(config, true_params={'pi': pi}),
            T=tree
        )

        print(f"Var Log-likelihood: {q.total_log_likelihood}")
        print(f"True Log-likelihood: {true_q.total_log_likelihood}")
        self.assertGreater(ari, 0.9, msg='ari less than 0.9. for K = 5')

