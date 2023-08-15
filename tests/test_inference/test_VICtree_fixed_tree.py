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
from variational_distributions.joint_dists import FixedTreeJointDist
from tests import model_variational_comparisons
from tests.utils_testing import simul_data_pyro_full_model, simulate_full_dataset_no_pyro
from utils import visualization_utils, data_handling
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

    def test_one_edge_tree(self):
        torch.manual_seed(0)
        logging.getLogger().setLevel("INFO")
        tree = tests.utils_testing.get_two_node_tree()
        n_nodes = len(tree.nodes)
        n_cells = 500
        n_sites = 500
        n_copy_states = 7
        dir_alpha = [1., 3.]
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 200.0

        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha)
        config = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3,
                        debug=False, diagnostics=False)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
        q.initialize()
        copy_tree = VICTree(config, q, y, draft=True)

        # Act
        copy_tree.run(n_iter=80)

        # Assert
        torch.set_printoptions(precision=2)
        out = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps,
                                                                q_c=copy_tree.q.c,
                                                                q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                                                                q_mt=copy_tree.q.mt)
        ari, perm, acc = (out['ari'], out['perm'], out['acc'])

        self.assertGreater(ari, 0.9, msg='ari less than 0.9 for easy scenario.')

    def test_three_node_tree(self):
        torch.manual_seed(0)
        tree = tests.utils_testing.get_tree_three_nodes_balanced()
        n_nodes = len(tree.nodes)
        n_cells = 500
        n_sites = 500
        n_copy_states = 7
        dir_alpha = [1., 3., 3.]
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 200.0
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha)
        print(f"Epsilon: {eps}")
        config = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3,
                        debug=False, diagnostics=False)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)

        q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
        # initialize all var dists
        q.initialize()
        qmt.update_params(mu=mu, lmbda=torch.ones(n_cells) * 10,
                          alpha=torch.ones(n_cells) * 10,
                          beta=torch.ones(n_cells) * 10)
        copy_tree = VICTree(config, q, y, draft=True)

        copy_tree.run(n_iter=50)

        # Assert
        torch.set_printoptions(precision=2)
        out = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps,
                                                                q_c=copy_tree.q.c,
                                                                q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                                                                q_mt=copy_tree.q.mt, q_eps=qeps)
        ari, perm, acc = (out['ari'], out['perm'], out['acc'])

        self.assertGreater(ari, 0.9, msg='ari less than 0.9 for easy scenario.')

    def test_five_node_tree(self):
        # CAN BE IGNORED IF TOO SLOW
        torch.manual_seed(0)
        K = 5
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 500
        n_sites = 500
        n_copy_states = 7
        dir_alpha0 = 10.
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 200.0
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha0)

        config = Config(n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3,
                        diagnostics=False, annealing=1.)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
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

        self.assertGreater(ari, 0.7, msg='ari less than 0.7.')

    def test_five_node_tree_local_vs_general_profiles(self):
        # CAN BE IGNORED IF TOO SLOW
        torch.manual_seed(0)
        K = 6
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        N = 500
        M_local = 100
        M_general = 2000
        A = 7
        dir_delta0 = 10.
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 100.0
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(N, M_local, A, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_delta0)

        config_local = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M_local, step_size=0.3,
                              diagnostics=False, annealing=1.)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config_local)
        q = FixedTreeJointDist(config_local, qc, qz, qeps, qmt, qpi, tree, y)
        q.initialize()
        victree = VICTree(config_local, q, y, draft=True)

        victree.run(n_iter=100)

        # Assert
        torch.set_printoptions(precision=2)
        out = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps,
                                                                q_c=victree.q.c,
                                                                q_z=victree.q.z, qpi=victree.q.pi,
                                                                q_mt=victree.q.mt, q_eps=qeps)

        # Extend clones with general structure
        M_tot = M_general + M_local
        tree_temp = nx.DiGraph()
        tree_temp.add_edge(0, 1)
        eps2, c2 = simul.simulate_copy_tree_data(2, M_general, A, tree_temp, eps_a=5., eps_b=M_tot, eps_0=0.01)
        c_general = c2[1].repeat(K, 1)  # Add same global structure to all clones
        c_general[0, :] = 2.  # reset root to 2
        y2 = simul.simulate_observations_Normal(N, M_general, c_general, z, mu, tau)
        eps_tot = {key: eps[key] * (M_local / M_tot) + eps2[(0, 1)] * (M_general / M_tot) for key in eps.keys()}
        c_tot = torch.zeros(K, M_tot)
        c_tot[:, 0:M_general] = c_general
        c_tot[:, M_general:M_tot] = C

        y_tot = torch.zeros((M_tot, N))
        y_tot[0:M_general, :] = y2
        y_tot[M_general:M_tot, :] = y

        config_general = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M_tot, step_size=0.3,
                                diagnostics=False, annealing=1.)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config_general, self.id().replace(".", "/"),
                                                                       base_dir='../../test_output')

        qc2, qt2, qeps2, qz2, qpi2, qmt2 = self.set_up_q(config_general)
        q2 = FixedTreeJointDist(config_general, qc2, qz2, qeps2, qmt2, qpi2, tree, y_tot)
        q2.initialize()
        victree2 = VICTree(config_general, q2, y_tot, draft=True)

        victree2.run(n_iter=100)

        # Assert
        torch.set_printoptions(precision=2)
        out = model_variational_comparisons.fixed_T_comparisons(obs=y_tot, true_C=c_tot, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps_tot,
                                                                q_c=victree2.q.c,
                                                                q_z=victree2.q.z, qpi=victree2.q.pi,
                                                                q_mt=victree2.q.mt, q_eps=victree2.q.eps)
        ari, perm, acc = (out['ari'], out['perm'], out['acc'])

        c_true_and_qc_viterbi = np.zeros((2, K, M_tot))
        c_true_and_qc_viterbi[0] = np.array(c_tot)
        c_true_and_qc_viterbi[1] = np.array(victree2.q.c.get_viterbi())
        visualization_utils.visualize_copy_number_profiles_of_multiple_sources(c_true_and_qc_viterbi,
                                                                               save_path=test_dir_name +
                                                                                         '/c_plot.png')
        visualization_utils.visualize_observations_copy_number_profiles_of_multiple_sources(c_true_and_qc_viterbi,
                                                                                            y_tot, z,
                                                                                            save_path=test_dir_name +
                                                                                                      '/c_obs_plot.png')

        # self.assertGreater(ari, 0.7, msg='ari less than 0.7.')

    def test_large_tree_fixed_qMuTau_same_data_different_optimizations(self):
        logger = logging.getLogger()
        logger.level = logging.INFO
        K = 7
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        print(f"Tree edges: {tree.edges}")
        n_cells = 500
        n_sites = 500
        n_copy_states = 7
        dir_alpha0 = [3.] * 7
        n_tests = 1
        config = Config(n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3,
                        debug=False)
        sim_data_seed = 0
        torch.manual_seed(sim_data_seed)
        out_simul = simul.simulate_full_dataset(config=config, eps_a=1.0, eps_b=20., mu0=1., lambda0=10., alpha0=50.,
                                                beta0=10., dir_delta=dir_alpha0, tree=tree)
        y = out_simul['obs']
        C = out_simul['c']
        z = out_simul['z']
        pi = out_simul['pi']
        mu = out_simul['mu']
        tau = out_simul['tau']
        eps = out_simul['eps']
        eps0 = out_simul['eps0']
        tree = out_simul['tree']

        print(f"Simulated data")
        vis_clone_idx = z[80]
        print(f"C: {C[vis_clone_idx, 40]} y: {y[80, 40]} z: {z[80]} \n"
              f"pi: {pi} mu: {mu[80]} tau: {tau[80]} eps: {eps}")

        for i in range(n_tests):
            torch.manual_seed(i)
            config = Config(n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3)
            qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
            qmt = qMuTau(config, true_params={"mu": mu, "tau": tau})
            q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
            q.initialize()
            qmt.initialize(method='fixed', loc=1., precision_factor=10., shape=50., rate=10.)
            q.eps.initialize(method='non_mutation')
            q.z.pi = f.one_hot(z.long(), num_classes=K).float()

            victree = VICTree(config, q, y, draft=True)

            victree.run(n_iter=50)

            torch.set_printoptions(precision=2)
            out = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                                    true_tau=tau, true_epsilon=eps, q_c=victree.q.c,
                                                                    q_z=victree.q.z, qpi=victree.q.pi,
                                                                    q_mt=victree.q.mt, q_eps=qeps)
            self.assertGreater(out['ari'], 0.9, msg='ARI score for qMuTau fixed to true values should be close to 1.')
            self.assertLess(out['qC_n_diff'], (n_sites * K) / 5,
                            msg='Number of wrong qC for qMuTau fixed to true values should be less than 20%.')

    @unittest.skip("long exec time")
    def test_large_tree_init_close_to_true_params(self):
        logger = logging.getLogger()
        logger.level = logging.INFO
        utils.config.set_seed(0)
        K = 5
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 1000
        n_sites = 200
        n_copy_states = 7
        nu_0 = 10.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 200.0
        dir_alpha0 = list(np.ones(K) * 2.)
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha0)
        config = Config(n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3,
                        diagnostics=True)
        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
        q.initialize()

        copy_tree = VICTree(config, q, y, draft=True)
        copy_tree.q.pi.concentration_param = dir_alpha0

        # Init Z with off-set from true params
        z_one_hot = f.one_hot(z, num_classes=K)
        off_set_z = 0.1
        z_perturbed = z_one_hot + off_set_z
        copy_tree.q.z.pi[...] = z_perturbed / z_perturbed.sum(1, keepdims=True)
        print(f'q(Z) init offset: {off_set_z * 100}%')

        # Init single filter probs with off-set from true params
        c_one_hot = f.one_hot(C.long(), num_classes=n_copy_states).float()
        off_set_c = 0.1
        c_perturbed = c_one_hot + off_set_c
        copy_tree.q.c.single_filtering_probs[...] = c_perturbed / c_perturbed.sum(dim=-1, keepdims=True)
        print(f'q(C) marginals init offset: {off_set_z * 100}%')

        # Init single filter probs with off-set from true params
        off_set_mutau = 0.1
        mu_perturbed = mu + mu * off_set_mutau
        tau_perturbed = tau + tau * off_set_mutau
        alpha_perturbed = torch.ones(n_cells) * 50.
        beta_perturbed = alpha_perturbed / tau_perturbed
        copy_tree.q.mt.nu[...] = mu_perturbed
        copy_tree.q.mt.alpha[...] = alpha_perturbed
        copy_tree.q.mt.beta[...] = beta_perturbed
        print(f'q(mu, tau) param init offset: {off_set_mutau * 100}%')

        # Act
        copy_tree.run(n_iter=50)

        # Assert
        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt,
                                                          q_eps=copy_tree.q.eps)

    @unittest.skip("long exec time")
    def test_large_tree_good_init_multiple_runs(self):
        logger = logging.getLogger()
        logger.level = logging.INFO
        K = 5
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 2000
        n_sites_list = [100, 100, 100, 100, 100]
        n_copy_states = 7
        dir_alpha0 = 1.
        n_tests = len(n_sites_list)
        alpha_0_list = [1., 1., 1.]
        beta_0_list = [1., 1., 1.]
        mu_0_list = [10., 10., 10., 10., 10.]
        lmbda_0_list = [10., 10., 10.]
        for i in range(n_tests):
            torch.manual_seed(i)
            print(f"---------- Experiment number {i} - seed {i} -----------")
            n_sites = n_sites_list[i]
            data = torch.ones((n_sites, n_cells))
            # FIXME: use simulate_full_dataset
            C, y, z, pi, mu, tau, eps = simul_data_pyro_full_model(data, n_cells, n_sites, n_copy_states, tree,
                                                                   mu_0=torch.tensor(mu_0_list[i]),
                                                                   lambda_0=1.,
                                                                   alpha0=10.,
                                                                   beta0=40.,
                                                                   a0=1.0,
                                                                   b0=10.0,
                                                                   dir_alpha0=1.0)

            visualization_utils.visualize_copy_number_profiles(C)
            config = Config(n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3)
            qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
            q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
            q.initialize()

            copy_tree = VICTree(config, q, y, draft=True)
            # copy_tree.q.pi.concentration_param = dir_alpha0 * torch.ones(K)
            z_one_hot = f.one_hot(z, num_classes=K)
            off_set_z = 0.2
            z_perturbed = z_one_hot + off_set_z
            copy_tree.q.z.pi[...] = z_perturbed / z_perturbed.sum(1, keepdims=True)

            c_one_hot = f.one_hot(C.long(), num_classes=n_copy_states).float()
            off_set_c = 0.0
            c_perturbed = c_one_hot + off_set_c
            copy_tree.q.c.single_filtering_probs[...] = c_perturbed / c_perturbed.sum(dim=-1, keepdims=True)

            copy_tree.run(n_iter=50)

            torch.set_printoptions(precision=2)
            model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                              true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                              q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                                                              q_mt=copy_tree.q.mt)

    @unittest.skip("long exec time")
    def test_large_tree_good_init_seiving(self):
        K = 5
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 1000
        n_sites_list = [100, 100, 100, 100, 100]
        n_copy_states = 7
        dir_alpha0 = 1.
        n_tests = len(n_sites_list)
        alpha_0_list = [1., 1., 1.]
        beta_0_list = [1., 1., 1.]
        mu_0_list = [10., 10., 10., 10., 10.]
        lmbda_0_list = [10., 10., 10.]
        for i in range(n_tests):
            torch.manual_seed(i)
            print(f"---------- Experiment number {i} - seed {i} -----------")
            n_sites = n_sites_list[i]
            data = torch.ones((n_sites, n_cells))
            # FIXME: sim
            C, y, z, pi, mu, tau, eps = simul_data_pyro_full_model(data, n_cells, n_sites, n_copy_states, tree,
                                                                   mu_0=mu_0_list[i],
                                                                   lambda_0=1.,
                                                                   alpha0=10.,
                                                                   beta0=40.,
                                                                   a0=1.0,
                                                                   b0=20.0,
                                                                   dir_alpha0=1.0)

            config = Config(n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, sieving_size=10,
                            n_sieving_iter=40, step_size=0.3)
            qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
            q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
            q.initialize()

            copy_tree = VICTree(config, q, y, draft=True)
            # copy_tree.q.pi.concentration_param = dir_alpha0 * torch.ones(K)
            z_one_hot = f.one_hot(z, num_classes=K)
            off_set_z = 0.2
            z_perturbed = z_one_hot + off_set_z
            copy_tree.q.z.pi[...] = z_perturbed / z_perturbed.sum(1, keepdims=True)

            c_one_hot = f.one_hot(C.long(), num_classes=n_copy_states).float()
            off_set_c = 0.0
            c_perturbed = c_one_hot + off_set_c
            copy_tree.q.c.single_filtering_probs[...] = c_perturbed / c_perturbed.sum(dim=-1, keepdims=True)

            copy_tree.run(n_iter=50)

            torch.set_printoptions(precision=2)
            model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                              true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                              q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                                                              q_mt=copy_tree.q.mt)
