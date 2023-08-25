import logging
import os
import random
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
from sklearn.metrics import adjusted_rand_score

import simul
import tests.utils_testing
import utils.config
from inference.victree import VICTree
from utils.data_handling import DataHandler
from variational_distributions.joint_dists import FixedTreeJointDist
from tests import model_variational_comparisons
from tests.utils_testing import simulate_full_dataset_no_pyro
from utils import visualization_utils, analysis_utils
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent, \
    qCMultiChrom


class VICTreeFixedTreeExperiment():
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

    def ari_as_function_of_K_experiment(self, save_plot=False, n_iter=500):
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
            y, C, z, pi, mu, tau, eps, eps0, adata = simulate_full_dataset_no_pyro(N, M, A, tree,
                                                                                   nu_0=nu_0,
                                                                                   lambda_0=lambda_0, alpha0=alpha0,
                                                                                   beta0=beta0,
                                                                                   a0=a0, b0=b0, dir_alpha0=dir_alpha0,
                                                                                   return_anndata=True)
            print(f"------------ Data set sanity check ------------")
            print(f"C: {C}")
            print(f"pi: {pi}")
            print(f"eps: {eps}")
            ari = []
            for seed in seeds:
                utils.config.set_seed(seed)
                config = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M, step_size=0.05,
                                save_progress_every_niter=n_iter + 1)
                qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
                q = FixedTreeJointDist(y, config, qc, qz, qeps, qmt, qpi, tree)
                q.initialize()
                dh = DataHandler(adata=adata)
                copy_tree = VICTree(config, q, y, dh)

                copy_tree.run(n_iter=n_iter)

                ari_seed = adjusted_rand_score(z, copy_tree.q.z.pi.argmax(dim=-1))
                ari.append(ari_seed)
                print(f"ARI for K {K} and seed {seed}: {ari_seed}")

            ari_list.append(ari)
            print(f"mean ARI for K {K}: {np.array(ari).mean()} ({np.array(ari).std()})")
            # Assert
            # model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
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
            base_dir = '../../../tests/test_output'
            test_dir_name = tests.utils_testing.create_experiment_output_catalog(path, base_dir)
            plt.savefig(test_dir_name + f"/ari_plot_N{N}_M{M}_A{A}.png")
            df = pd.DataFrame({
                'm': K_list,
                'ari': ari_means,
                'sd': ari_stds
            })
            df.to_csv(os.path.join(test_dir_name, f"k_ari_N{N}_M{M}_A{A}.csv"), index=False)

    def ari_as_function_of_M_experiment(self, save_plot=False, n_iter=500):
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
            y, C, z, pi, mu, tau, eps, eps0, adata = simulate_full_dataset_no_pyro(N, M, A, tree,
                                                                                   nu_0=nu_0,
                                                                                   lambda_0=lambda_0, alpha0=alpha0,
                                                                                   beta0=beta0,
                                                                                   a0=a0, b0=b0, dir_alpha0=dir_alpha0,
                                                                                   return_anndata=True)
            print(f"------------ Data set sanity check ------------")
            print(f"C: {C}")
            print(f"pi: {pi}")
            ari = []
            for seed in seeds:
                utils.config.set_seed(seed)
                config = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M, step_size=0.1,
                                save_progress_every_niter=n_iter + 1)
                qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
                q = FixedTreeJointDist(y, config, qc, qz, qeps, qmt, qpi, tree)
                q.initialize()
                dh = DataHandler(adata=adata)
                copy_tree = VICTree(config, q, y, dh)

                copy_tree.run(n_iter=n_iter)

                ari_seed = adjusted_rand_score(z, copy_tree.q.z.pi.argmax(dim=-1))
                ari.append(ari_seed)
                print(f"ARI for M {M} and seed {seed}: {ari_seed}")

            ari_list.append(ari)
            # Assert
            # model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
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
            base_dir = '../../../tests/test_output'
            test_dir_name = tests.utils_testing.create_experiment_output_catalog(path, base_dir)
            plt.savefig(test_dir_name + f"/ari_plot_N{N}_K{K}_A{A}.png")
            df = pd.DataFrame({
                'm': M_list,
                'ari': ari_means,
                'sd': ari_stds
            })
            df.to_csv(os.path.join(test_dir_name, f"m_ari_N{N}_K{K}_A{A}.csv"), index=False)

    def fixed_tree_real_data_experiment(self, save_plot=False, n_iter=500):
        # Hyper parameters
        seeds = list(range(0, 5))
        K = 7
        A = 7
        step_size = 1.0

        # Prior parameters
        dir_alpha0 = 10.
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 1.0
        b0 = 200.0

        # Fixed tree
        tree = nx.DiGraph()
        tree.add_edge(0, 1)
        tree.add_edge(0, 2)
        tree.add_edge(1, 3)
        tree.add_edge(1, 4)
        tree.add_edge(3, 5)
        tree.add_edge(3, 6)

        # Load data
        file_path = './../../../data/x_data/P01-066_cn_data.h5ad'
        data_handler = DataHandler(file_path)
        y = data_handler.norm_reads
        M, N = y.shape

        # Output storage variables
        elbo_list = []
        elbo = []

        if save_plot:
            dirs = os.getcwd().split('/')
            dir_top_idx = dirs.index('experiments')
            dir_path = dirs[dir_top_idx:]
            path = os.path.join(*dir_path, self.__class__.__name__, sys._getframe().f_code.co_name)
            path = os.path.join(path, f'K{K}_A{A}_rho{step_size}_niter{n_iter}')
            base_dir = '../../../tests/test_output'
            test_dir_name = tests.utils_testing.create_experiment_output_catalog(path, base_dir)

        for seed in seeds:
            utils.config.set_seed(seed)
            config = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M, step_size=step_size,
                            save_progress_every_niter=n_iter + 1, chromosome_indexes=data_handler.get_chr_idx(),
                            out_dir=test_dir_name, split=True)
            qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
            qeps = qEpsilonMulti(config, alpha_prior=a0, beta_prior=b0)
            qpi = qPi(config, delta_prior=dir_alpha0)
            qmt = qMuTau(config, nu_prior=nu_0, lambda_prior=lambda_0, alpha_prior=alpha0, beta_prior=beta0)
            qc = qCMultiChrom(config)
            q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
            q.initialize()
            qmt.initialize(method='prior')
            victree = VICTree(config, q, y, data_handler)

            victree.run(n_iter=n_iter)

            elbo_seed = victree.elbo
            elbo.append(elbo_seed)
            print(f"ARI for M {M} and seed {seed}: {elbo_seed}")
            if save_plot:
                analysis_utils.write_inference_output_no_ground_truth(victree, y, test_dir_name, f'seed{seed}_',
                                                                      tree=tree)

        np_elbo = np.array(elbo)

        if save_plot:
            df = pd.DataFrame({
                'elbo': np_elbo,
            })
            df.to_csv(os.path.join(test_dir_name, f"realdata_ELBO_K{K}_A{A}.csv"), index=False)


if __name__ == '__main__':
    n_iter = 200
    experiment_class = VICTreeFixedTreeExperiment()
    experiment_class.fixed_tree_real_data_experiment(save_plot=True, n_iter=n_iter)
    #experiment_class.ari_as_function_of_K_experiment(save_plot=True, n_iter=n_iter)
    #experiment_class.ari_as_function_of_M_experiment(save_plot=True, n_iter=n_iter)
