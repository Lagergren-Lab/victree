import logging
import os
import random
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as f
from sklearn.metrics import adjusted_rand_score

import simul
import tests.utils_testing
import utils.config
from analysis import qC_analysis
from inference.victree import VICTree
from variational_distributions.joint_dists import FixedTreeJointDist, VarTreeJointDist
from tests import model_variational_comparisons
from tests.utils_testing import simul_data_pyro_full_model, simulate_full_dataset_no_pyro
from utils import visualization_utils, analysis_utils, tree_utils
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent


class TuneModelToFixedTreeExperiment():
    """
    Test class for running small scale, i.e. runnable on local machine, experiments for fixed trees.
    Not using unittest test framework as it is incompatible with matplotlib GUI-backend.
    """

    def set_up_q(self, config):
        qc = qC(config)
        qt = qT(config)
        qeps = qEpsilonMulti(config, alpha_prior=1., beta_prior=20.)
        qz = qZ(config)
        qpi = qPi(config, delta_prior=100.)
        qmt = qMuTau(config, lambda_prior=10., alpha_prior=5000., beta_prior=500.)
        return qc, qt, qeps, qz, qpi, qmt

    def set_up_q_fixed_T(self, config):
        qc = qC(config)
        qeps = qEpsilonMulti(config)
        qz = qZ(config)
        qpi = qPi(config)
        qmt = qMuTau(config)
        return qc, qeps, qz, qpi, qmt


    def simulate_data(self, N, M, K, A):
        utils.config.set_seed(0)

        dir_alpha0 = 10.
        nu_0 = torch.tensor(1.)
        lambda_0 = torch.tensor(10.)
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        tree = tests.utils_testing.get_tree_K_nodes_random(K)

        a0 = torch.tensor(1.0)
        b0 = torch.tensor(200.0)
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(N, M, A, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0,
                                                                        beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha0)
        print(f"------------ Data set sanity check ------------")
        print(f"C: {C}")
        print(f"pi: {pi}")
        return y, C, z, pi, mu, tau, eps, eps0, tree

    def train_model_vs_fixed_tree_post_inference_experiment(self, save_plot=False):
        output_dir = "./../../test_output/experiments/tune_model_to_fixed_tree_experiment"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        N = 1000
        M = 4000
        K = 8
        A = 7
        y, c, z, pi, mu, tau, eps, eps0, tree = self.simulate_data(N, M, K, A)
        ari_list = []
        seeds = list(range(0, 1))


        ari = []
        for seed in seeds:
            utils.config.set_seed(seed)
            config = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M,
                            step_size=0.3, out_dir=output_dir, wis_sample_size=5)
            qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
            q = VarTreeJointDist(config, y, qc, qz, qt, qeps, qmt, qpi)
            q.initialize()
            victree = VICTree(config, q, y)

            victree.run(n_iter=30)

            ari_seed = adjusted_rand_score(z, victree.q.z.pi.argmax(dim=-1))
            ari.append(ari_seed)
            print(f"ARI for M {M} and seed {seed}: {ari_seed}")
            ari_list.append(ari)


            # Assert
            out_comparisons = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=c, true_Z=z, true_pi=pi, true_mu=mu,
                                                              true_tau=tau, true_epsilon=eps, q_c=victree.q.c,
                                                              q_z=victree.q.z, qpi=victree.q.pi,
                                                              q_mt=victree.q.mt)
            best_perm = out_comparisons['perm']
            G = victree.q.t.weighted_graph
            T = nx.maximum_spanning_arborescence(G)
            print(f'Edmonds tree of W: {tree_utils.relabel_nodes(T, best_perm).edges}')
            print(f'True tree: {tree.edges}')

            c_all = np.zeros((3, K, M))
            qC_marg_remapped = analysis_utils.remap_tensor(victree.q.c.single_filtering_probs.argmax(dim=-1), best_perm)

            victree_fixed_T = qC_analysis.train_on_fixed_tree(victree=victree, n_iter=50)
            model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=c, true_Z=z, true_pi=pi, true_mu=mu,
                                                              true_tau=tau, true_epsilon=eps, q_c=victree_fixed_T.q.c,
                                                              q_z=victree_fixed_T.q.z, qpi=victree_fixed_T.q.pi,
                                                              q_mt=victree_fixed_T.q.mt)



        qC_marg_remapped_fixed_T = analysis_utils.remap_tensor(victree_fixed_T.q.c.single_filtering_probs.argmax(dim=-1), best_perm)
        c_all[0] = np.array(c)
        c_all[1] = np.array(qC_marg_remapped)  # remap after best_perm
        c_all[2] = np.array(qC_marg_remapped_fixed_T)
        visualization_utils.visualize_copy_number_profiles_of_multiple_sources(c_all)


if __name__ == '__main__':
    experiment_class = TuneModelToFixedTreeExperiment()
    experiment_class.train_model_vs_fixed_tree_post_inference_experiment(save_plot=False)