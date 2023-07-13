import logging
import os
import random
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as f
from pyro import poutine
from sklearn.metrics import adjusted_rand_score

import simul
import tests.utils_testing
import utils.config
from inference.victree import VICTree
from variational_distributions.joint_dists import FixedTreeJointDist
from tests import model_variational_comparisons
from tests.utils_testing import simul_data_pyro_full_model, simulate_full_dataset_no_pyro
from utils import visualization_utils
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent


class VICtreeFixedTreeExperiment():
    """
    Test class for running small scale, i.e. runnable on local machine, experiments for fixed trees.
    Not using unittest test framework as it is incompatible with matplotlib GUI-backend.
    """

    def set_up_q(self, config):
        qc = qC(config)
        qt = qT(config)
        qeps = qEpsilonMulti(config)
        qz = qZ(config)
        qpi = qPi(config)
        qmt = qMuTau(config)
        return qc, qt, qeps, qz, qpi, qmt

    def ari_as_function_of_K_experiment(self, save_plot=False):
        utils.config.set_seed(0)

        K_list = list(range(3, 16))
        ari_list = []
        seeds = list(range(0, 5))

        N = 500
        M = 3000
        A = 7
        dir_alpha0 = 10.
        nu_0 = torch.tensor(1.)
        lambda_0 = torch.tensor(10.)
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)

        for K in K_list:
            tree = tests.utils_testing.get_tree_K_nodes_random(K)

            a0 = torch.tensor(5.0)
            b0 = torch.tensor(200.0)
            y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(N, M, A, tree,
                                                                            nu_0=nu_0,
                                                                            lambda_0=lambda_0, alpha0=alpha0,
                                                                            beta0=beta0,
                                                                            a0=a0, b0=b0, dir_alpha0=dir_alpha0)
            print(f"------------ Data set sanity check ------------")
            print(f"C: {C}")
            print(f"pi: {pi}")
            print(f"eps: {eps}")
            ari = []
            for seed in seeds:
                utils.config.set_seed(seed)
                config = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M, step_size=0.05)
                qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
                q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
                q.initialize()
                copy_tree = VICTree(config, q, y)

                copy_tree.run(n_iter=500)

                ari_seed = adjusted_rand_score(z, copy_tree.q.z.pi.argmax(dim=-1))
                ari.append(ari_seed)
                print(f"ARI for K {K} and seed {seed}: {ari_seed}")

            ari_list.append(ari)
            print(f"mean ARI for K {K}: {np.array(ari).mean()} ({np.array(ari).std()})")
            # Assert
                #model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                #                                                  true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                #                                                  q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                #                                                  q_mt=copy_tree.q.mt)

        np_ari = np.array(ari_list)
        ari_means = np_ari.mean(axis=-1)
        ari_stds = np_ari.std(axis=-1)
        plt.errorbar(K_list, ari_means, ari_stds, linestyle='None', marker='^')
        plt.xlabel('K - number of clones')
        plt.ylabel('ARI score')
        plt.title(f'Fixed tree clustering performance - N: {N} - M: {M} - A: {A}')
        if save_plot:
            dirs = os.getcwd().split('/')
            dir_top_idx = dirs.index('experiments')
            dir_path = dirs[dir_top_idx:]
            path = os.path.join(*dir_path, self.__class__.__name__, sys._getframe().f_code.co_name)
            base_dir = '../../test_output'
            test_dir_name = tests.utils_testing.create_experiment_output_catalog(path, base_dir)
            plt.savefig(test_dir_name + f"/ari_plot_N{N}_M{M}_A{A}.png")


    def ari_as_function_of_M_experiment(self, save_plot=False):
        utils.config.set_seed(0)

        M_list = [50, 100, 200, 500, 1000, 2000, 5000]
        ari_list = []
        seeds = list(range(0, 5))

        N = 1000
        K = 7
        A = 7
        dir_alpha0 = 10.
        nu_0 = torch.tensor(1.)
        lambda_0 = torch.tensor(10.)
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)

        for M in M_list:
            tree = tests.utils_testing.get_tree_K_nodes_random(K)

            a0 = torch.tensor(10.0)
            b0 = torch.tensor(200.0)
            y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(N, M, A, tree,
                                                                            nu_0=nu_0,
                                                                            lambda_0=lambda_0, alpha0=alpha0,
                                                                            beta0=beta0,
                                                                            a0=a0, b0=b0, dir_alpha0=dir_alpha0)
            print(f"------------ Data set sanity check ------------")
            print(f"C: {C}")
            print(f"pi: {pi}")
            ari = []
            for seed in seeds:
                utils.config.set_seed(seed)
                config = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M, step_size=0.1)
                qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
                q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
                q.initialize()
                copy_tree = VICTree(config, q, y)

                copy_tree.run(n_iter=500)

                ari_seed = adjusted_rand_score(z, copy_tree.q.z.pi.argmax(dim=-1))
                ari.append(ari_seed)
                print(f"ARI for M {M} and seed {seed}: {ari_seed}")

            ari_list.append(ari)
                # Assert
                #model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                #                                                  true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                #                                                  q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                #                                                  q_mt=copy_tree.q.mt)

        np_ari = np.array(ari_list)
        ari_means = np_ari.mean(axis=-1)
        ari_stds = np_ari.std(axis=-1)
        plt.errorbar(M_list, ari_means, ari_stds, linestyle='None', marker='^')
        plt.xlabel('M - number of bins')
        plt.ylabel('ARI score')
        plt.title(f'Fixed tree clustering performance - N: {N} - K: {K} - A: {A}')
        if save_plot:
            dirs = os.getcwd().split('/')
            dir_top_idx = dirs.index('experiments')
            dir_path = dirs[dir_top_idx:]
            path = os.path.join(*dir_path, self.__class__.__name__, sys._getframe().f_code.co_name)
            base_dir = '../../test_output'
            test_dir_name = tests.utils_testing.create_experiment_output_catalog(path, base_dir)
            plt.savefig(test_dir_name + f"/ari_plot_N{N}_K{K}_A{A}.png")


if __name__ == '__main__':
    experiment_class = VICtreeFixedTreeExperiment()
    experiment_class.ari_as_function_of_K_experiment(save_plot=True)