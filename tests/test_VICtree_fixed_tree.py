import logging
import random
import unittest

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as f
from pyro import poutine

import simul
import tests.utils_testing
from inference.copy_tree import VarDistFixedTree, CopyTree
from tests import model_variational_comparisons
from tests.utils_testing import simul_data_pyro_full_model, pyro_simulate_full_dataset
from utils import visualization_utils
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent


class VICtreeFixedTreeTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # build config
        self.config = Config(n_nodes=3, n_states=7, n_cells=20, chain_length=10, debug=True)
        self.qc = qC(self.config)
        self.qc.initialize()
        self.qt = qT(self.config)
        self.qeps = qEpsilonMulti(self.config, 2, 5)  # skewed towards 0
        self.qz = qZ(self.config)
        self.qpi = qPi(self.config)
        self.qmt = qMuTau(self.config)
        self.qmt.initialize(loc=1, precision_factor=.1, shape=1, rate=1)
        torch.manual_seed(0)
        random.seed(0)

    def set_up_q(self, config):
        qc = qC(config)
        qt = qT(config)
        qeps = qEpsilonMulti(config)
        qz = qZ(config)
        qpi = qPi(config)
        qmt = qMuTau(config)
        return qc, qt, qeps, qz, qpi, qmt

    def simul_data_pyro_fixed_parameters(self, data, n_cells, n_sites, n_copy_states, tree: nx.DiGraph,
                                         mu_0=torch.tensor(10.),
                                         lambda_0=torch.tensor(.1),
                                         alpha0=torch.tensor(10.),
                                         beta0=torch.tensor(40.),
                                         a0=torch.tensor(1.0),
                                         b0=torch.tensor(20.0),
                                         dir_alpha0=torch.tensor(1.0)
                                         ):
        model_tree_markov_full = simul.model_tree_markov_full
        unconditioned_model = poutine.uncondition(model_tree_markov_full)
        C, y, z, pi, mu, tau, eps = unconditioned_model(data, n_cells, n_sites, n_copy_states, tree,
                                                        mu_0,
                                                        lambda_0,
                                                        alpha0,
                                                        beta0,
                                                        a0,
                                                        b0,
                                                        dir_alpha0)

        return C, y, z, pi, mu, tau, eps

    def test_one_edge_tree(self):
        torch.manual_seed(0)
        logging.getLogger().setLevel("INFO")
        tree = tests.utils_testing.get_two_node_tree()
        n_nodes = len(tree.nodes)
        n_cells = 1000
        n_sites = 200
        n_copy_states = 7
        dir_alpha = torch.tensor([1., 3.])
        y, C, z, pi, mu, tau, eps, eps0 = pyro_simulate_full_dataset(n_cells, n_sites, n_copy_states, tree,
                                                                     mu_0=torch.tensor(10.),
                                                                     lambda_0=torch.tensor(10.),
                                                                     alpha0=torch.tensor(500.),
                                                                     beta0=torch.tensor(50.),
                                                                     a0=torch.tensor(1.0),
                                                                     b0=torch.tensor(20.0),
                                                                     dir_alpha0=dir_alpha
                                                                     )
        print(f"C node 1 site 20: {C[1, 20]}")
        print(f"Epsilon: {eps}")
        config = Config(step_size=0.3, n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites,
                        debug=False, diagnostics=True)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = VarDistFixedTree(config, qc, qz, qeps, qmt, qpi, tree, y)
        # initialize all var dists
        q.initialize()
        copy_tree = CopyTree(config, q, y)

        copy_tree.run(30)
        diagnostics_dict = copy_tree.diagnostics_dict
        visualization_utils.visualize_diagnostics(diagnostics_dict,
                                                  cells_to_vis_idxs=[0, int(n_cells/2), int(n_cells/3), n_cells-1])#,
                                                  #save_path='./test_output/one_edge_tree_test.png')
        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                                                          q_mt=copy_tree.q.mt)

    def test_three_node_tree(self):
        torch.manual_seed(2)
        tree = tests.utils_testing.get_tree_three_nodes_balanced()
        n_nodes = len(tree.nodes)
        n_cells = 1000
        n_sites = 200
        n_copy_states = 7
        dir_alpha = torch.tensor([1., 3., 3.])
        y, C, z, pi, mu, tau, eps, eps0 = pyro_simulate_full_dataset(n_cells, n_sites, n_copy_states, tree,
                                                                     mu_0=torch.tensor(10.),
                                                                     lambda_0=torch.tensor(10.),
                                                                     alpha0=torch.tensor(500.),
                                                                     beta0=torch.tensor(50.),
                                                                     a0=torch.tensor(1.0),
                                                                     b0=torch.tensor(20.0),
                                                                     dir_alpha0=dir_alpha
                                                                     )
        print(f"C node 1 site 2: {C[1, 2]}")
        print(f"Epsilon: {eps}")
        config = Config(step_size=0.3, n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites,
                        debug=False, diagnostics=True)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)

        q = VarDistFixedTree(config, qc, qz, qeps, qmt, qpi, tree, y)
        # initialize all var dists
        q.initialize()
        qmt.update_params(mu=mu, lmbda=torch.ones(n_cells) * 10,
                          alpha=torch.ones(n_cells) * 10,
                          beta=torch.ones(n_cells) * 10)
        copy_tree = CopyTree(config, q, y)

        copy_tree.run(50)

        torch.set_printoptions(precision=2)
        diagnostics_dict = copy_tree.diagnostics_dict
        visualization_utils.visualize_diagnostics(diagnostics_dict,
                                                  cells_to_vis_idxs=[0, int(n_cells / 2), int(n_cells / 3),
                                                                     n_cells - 1])
        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                                                          q_mt=copy_tree.q.mt)

    def test_large_tree(self):
        torch.manual_seed(0)
        K = 5
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 200
        n_sites = 100
        n_copy_states = 7
        data = torch.ones((n_sites, n_cells))
        dir_alpha0 = torch.ones(K)
        dir_alpha0[3] = 100.
        C, y, z, pi, mu, tau, eps = simul_data_pyro_full_model(data, n_cells, n_sites, n_copy_states, tree,
                                                               dir_alpha0=dir_alpha0)
        print(f"C node 1 site 2: {C[1, 2]}")
        config = Config(n_nodes=K, chain_length=n_sites,
                        n_cells=n_cells, n_states=n_copy_states)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = VarDistFixedTree(config, qc, qz, qeps, qmt, qpi, tree, y)
        copy_tree = CopyTree(config, q, y)

        copy_tree.run(10)
        q_C = copy_tree.q.c.single_filtering_probs
        q_z_pi = copy_tree.q.z.pi
        delta = copy_tree.q.pi.concentration_param
        print(f"True pi: {pi} \n variational pi_n: {q_z_pi[0:5]}")
        print(f"True alpha: {dir_alpha0} \n variational concentration param: {delta}")
        print(f"True C: {C[1, 5:10]} \n q(C): {q_C[1, 5:10, :]}")

    def test_large_tree_init_true_params(self):
        K = 5
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 1000
        n_sites = 100
        n_copy_states = 7
        data = torch.ones((n_sites, n_cells))
        dir_alpha0 = 1.
        C, y, z, pi, mu, tau, eps = simul_data_pyro_full_model(data, n_cells, n_sites, n_copy_states, tree,
                                                               dir_alpha0=dir_alpha0)
        config = Config(n_nodes=K, chain_length=n_sites, n_cells=n_cells, n_states=n_copy_states)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = VarDistFixedTree(config, qc, qz, qeps, qmt, qpi, tree, y)
        q.initialize(eps_alpha=10., eps_beta=40.,
                     loc=mu, precision_factor=.1, shape=5, rate=5)

        copy_tree = CopyTree(config, q, y)
        copy_tree.q.pi.concentration_param = dir_alpha0 * torch.ones(K)
        copy_tree.q.z.pi[...] = f.one_hot(z, num_classes=K)
        copy_tree.q.c.single_filtering_probs[...] = f.one_hot(C.long(), num_classes=n_copy_states).float()

        copy_tree.run(20)
        q_C = copy_tree.q.c.single_filtering_probs
        q_pi = copy_tree.q.z.pi
        delta = copy_tree.q.pi.concentration_param

        torch.set_printoptions(precision=2)
        verbose = False
        if verbose:
            print(f"Prior dirichlet param: {dir_alpha0 * torch.ones(K)} \n variational concentration param: {delta}")
            print(f"True pi: {pi} \n variational categorical param: {torch.mean(q_pi, dim=0)}")
            print(f"True C: {f.one_hot(C[1, 5:10].long(), num_classes=n_copy_states)} \n q(C): {q_C[1, 5:10, :]}")
            print(f"True C: {f.one_hot(C[3, 5:10].long(), num_classes=n_copy_states)} \n q(C): {q_C[3, 5:10, :]}")
            print(f"True C: {f.one_hot(C[4, 80:100].long(), num_classes=n_copy_states)} \n q(C): {q_C[4, 80:100, :]}")
            print(f"True mu: {mu[0:5]} \n Expected mu: {qmt.nu[0:5]}")
            print(f"True tau: {tau[0:5]} \n Expected var tau: {qmt.exp_tau()[0:5]}")

        print(f'tree: {tree.edges}')

        # Asserts
        print(f"Number of differing argmax(qC) and true C: {torch.sum(C != torch.argmax(q_C, dim=-1))}")
        threshold = 10
        n_diff_qC_vs_true_C = torch.sum(C != torch.argmax(q_C, dim=-1))
        self.assertTrue(n_diff_qC_vs_true_C < threshold, msg=f"N diff argmax q(C) vs true C: {n_diff_qC_vs_true_C}")
        self.assertTrue(torch.allclose(pi, torch.mean(q_pi, dim=0), atol=0.1),
                        msg=f"q(Z) mean over cells not close to true pi used for sampling Z."
                            f" \n q(Z)) mean: {torch.mean(q_pi, dim=0)} \n true pi: {pi}")

    def test_large_tree_init_true_params_multiple_runs(self):
        logger = logging.getLogger()
        logger.level = logging.INFO
        K = 10
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 1000
        n_sites_list = [10, 100, 300]
        n_copy_states = 7
        dir_alpha0 = 1.
        n_tests = 3
        alpha_0_list = [1., 1., 1.]
        beta_0_list = [1., 1., 1.]
        mu_0_list = [10., 10., 10.]
        lmbda_0_list = [10., 10., 10.]
        for i in range(n_tests):
            torch.manual_seed(i)
            n_sites = n_sites_list[i]
            data = torch.ones((n_sites, n_cells))
            C, y, z, pi, mu, tau, eps = simul_data_pyro_full_model(data, n_cells, n_sites, n_copy_states, tree,
                                                                   mu_0=torch.tensor(mu_0_list[i]),
                                                                   lambda_0=torch.tensor(1.),
                                                                   alpha0=torch.tensor(10.),
                                                                   beta0=torch.tensor(40.),
                                                                   a0=torch.tensor(1.0),
                                                                   b0=torch.tensor(20.0),
                                                                   dir_alpha0=torch.tensor(1.0))

            config = Config(n_nodes=K, chain_length=n_sites, n_cells=n_cells, n_states=n_copy_states)
            qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
            q = VarDistFixedTree(config, qc, qz, qeps, qmt, qpi, tree, y)
            q.initialize(eps_alpha=1., eps_beta=20.,
                         loc=mu, precision_factor=.1, shape=5, rate=5)

            copy_tree = CopyTree(config, q, y)
            copy_tree.q.pi.concentration_param = dir_alpha0 * torch.ones(K)
            copy_tree.q.z.pi[...] = f.one_hot(z, num_classes=K)
            copy_tree.q.c.single_filtering_probs[...] = f.one_hot(C.long(), num_classes=n_copy_states).float()

            copy_tree.run(20)
            q_C = copy_tree.q.c.single_filtering_probs
            q_pi = copy_tree.q.z.pi
            delta = copy_tree.q.pi.concentration_param

            torch.set_printoptions(precision=2)
            model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                              true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                              q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                                                              q_mt=copy_tree.q.mt)

    def test_large_tree_good_init_multiple_runs(self):
        logger = logging.getLogger()
        logger.level = logging.INFO
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
            C, y, z, pi, mu, tau, eps = simul_data_pyro_full_model(data, n_cells, n_sites, n_copy_states, tree,
                                                                   mu_0=torch.tensor(mu_0_list[i]),
                                                                   lambda_0=torch.tensor(1.),
                                                                   alpha0=torch.tensor(10.),
                                                                   beta0=torch.tensor(40.),
                                                                   a0=torch.tensor(1.0),
                                                                   b0=torch.tensor(10.0),
                                                                   dir_alpha0=torch.tensor(1.0))

            visualization_utils.visualize_copy_number_profiles(C)
            config = Config(step_size=0.3, n_nodes=K, chain_length=n_sites, n_cells=n_cells, n_states=n_copy_states)
            qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
            q = VarDistFixedTree(config, qc, qz, qeps, qmt, qpi, tree, y)
            q.initialize()

            copy_tree = CopyTree(config, q, y)
            # copy_tree.q.pi.concentration_param = dir_alpha0 * torch.ones(K)
            z_one_hot = f.one_hot(z, num_classes=K)
            off_set_z = 0.2
            z_perturbed = z_one_hot + off_set_z
            copy_tree.q.z.pi[...] = z_perturbed / z_perturbed.sum(1, keepdims=True)

            c_one_hot = f.one_hot(C.long(), num_classes=n_copy_states).float()
            off_set_c = 0.0
            c_perturbed = c_one_hot + off_set_c
            copy_tree.q.c.single_filtering_probs[...] = c_perturbed / c_perturbed.sum(dim=-1, keepdims=True)

            copy_tree.run(50)
            q_C = copy_tree.q.c.single_filtering_probs
            q_pi = copy_tree.q.z.pi
            delta = copy_tree.q.pi.concentration_param

            torch.set_printoptions(precision=2)
            model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                              true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                              q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                                                              q_mt=copy_tree.q.mt)

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
            C, y, z, pi, mu, tau, eps = simul_data_pyro_full_model(data, n_cells, n_sites, n_copy_states, tree,
                                                                   mu_0=torch.tensor(mu_0_list[i]),
                                                                   lambda_0=torch.tensor(1.),
                                                                   alpha0=torch.tensor(10.),
                                                                   beta0=torch.tensor(40.),
                                                                   a0=torch.tensor(1.0),
                                                                   b0=torch.tensor(20.0),
                                                                   dir_alpha0=torch.tensor(1.0))

            config = Config(step_size=0.3,
                            sieving_size=10,
                            n_sieving_runs=40,
                            n_nodes=K,
                            chain_length=n_sites,
                            n_cells=n_cells,
                            n_states=n_copy_states)
            qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
            q = VarDistFixedTree(config, qc, qz, qeps, qmt, qpi, tree, y)
            q.initialize(eps_alpha=10., eps_beta=40.,
                         loc=mu, precision_factor=.1, shape=5, rate=5)

            copy_tree = CopyTree(config, q, y)
            # copy_tree.q.pi.concentration_param = dir_alpha0 * torch.ones(K)
            z_one_hot = f.one_hot(z, num_classes=K)
            off_set_z = 0.2
            z_perturbed = z_one_hot + off_set_z
            copy_tree.q.z.pi[...] = z_perturbed / z_perturbed.sum(1, keepdims=True)

            c_one_hot = f.one_hot(C.long(), num_classes=n_copy_states).float()
            off_set_c = 0.0
            c_perturbed = c_one_hot + off_set_c
            copy_tree.q.c.single_filtering_probs[...] = c_perturbed / c_perturbed.sum(dim=-1, keepdims=True)

            copy_tree.run(50)

            torch.set_printoptions(precision=2)
            model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                              true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                              q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                                                              q_mt=copy_tree.q.mt)
