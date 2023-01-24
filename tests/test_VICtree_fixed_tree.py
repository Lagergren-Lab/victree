import random
import unittest

import networkx as nx
import torch
import torch.nn.functional as f
from pyro import poutine

import simul
import tests.utils_testing
from inference.copy_tree import VarDistFixedTree, CopyTree
from model.generative_model import GenerativeModel
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
        qmt.initialize(loc=1, precision_factor=1., shape=10, rate=1)
        return qc, qt, qeps, qz, qpi, qmt

    def simul_data_pyro_full_model(self, data, n_cells, n_sites, n_copy_states, tree: nx.DiGraph,
                                   mu_0=torch.tensor(1.),
                                   lambda_0=torch.tensor(1.),
                                   alpha0=torch.tensor(1.),
                                   beta0=torch.tensor(1.),
                                   a0=torch.tensor(1.0),
                                   b0=torch.tensor(10.0),
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

    def simul_data_pyro_fixed_parameters(self, data, n_cells, n_sites, n_copy_states, tree: nx.DiGraph,
                        mu_0=torch.tensor(10.),
                        lambda_0=torch.tensor(.1),
                        alpha0=torch.tensor(10.),
                        beta0=torch.tensor(40.),
                        a0=torch.tensor(1.0),
                        b0=torch.tensor(1.0),
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

    def test_small_tree(self):
        torch.manual_seed(0)
        tree = tests.utils_testing.get_tree_three_nodes_balanced()
        n_nodes = len(tree.nodes)
        n_cells = 100
        n_sites = 50
        n_copy_states = 7
        data = torch.ones((n_sites, n_cells))
        dir_alpha = torch.tensor([3., 3., 3.])
        C, y, z, pi, mu, tau, eps = self.simul_data_pyro_full_model(data, n_cells, n_sites, n_copy_states, tree,
                                                                    mu_0=10., dir_alpha0=dir_alpha)
        # y should be integer and non-negative (count data)
        # y = y.clamp(min=0).int()
        print(f"C node 1 site 2: {C[1, 2]}")
        print(f"Epsilon: {eps}")
        config = Config(step_size=1., n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites,
                        debug=False)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        qmt = qMuAndTauCellIndependent(config)

        p = GenerativeModel(config, tree)
        q = VarDistFixedTree(config, qc, qz, qeps, qmt, qpi, tree, y)
        # initialize all var dists
        q.initialize(loc=1, precision_factor=.1, shape=5, rate=5)
        qmt.update_params(mu=mu, lmbda=torch.ones(n_cells) * 1,
                          alpha=torch.ones(n_cells) * 1,
                          beta=torch.ones(n_cells) * 1)
        copy_tree = CopyTree(config, p, q, y)

        copy_tree.run(50)

        q_C = copy_tree.q.c.single_filtering_probs
        q_z_pi = copy_tree.q.z.pi
        torch.set_printoptions(precision=2)
        q_eps = q.eps
        q_mt = q.mt
        print(f"q_epsilon mean: {q_eps.alpha / q_eps.beta}")
        print(f"True Z: {z[0:10]} \n variational pi_n: {q_z_pi[0:10]}")
        print(f"True mu: {mu[0:10]} \n E_q[mu_n]: {q_mt.nu[0:10]}")
        print(f"y_mn: {y[0:10, 0:10]}")
        print(f"True C: {C[1, 5:10]} \n q(C): {q_C[1, 5:10, :]}")
        print(f"True C: {C[1, 45:50]} \n q(C): {q_C[1, 45:50, :]}")
        print(f"True C: {C[2, 5:10]} \n q(C): {q_C[2, 5:10, :]}")
        print(f"True C: {C[2, 45:50]} \n q(C): {q_C[2, 45:50, :]}")

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
        C, y, z, pi, mu, tau, eps = self.simul_data_pyro_full_model(data, n_cells, n_sites, n_copy_states, tree,
                                                                    dir_alpha0=dir_alpha0)
        print(f"C node 1 site 2: {C[1, 2]}")
        config = Config(n_nodes=K, chain_length=n_sites,
                        n_cells=n_cells, n_states=n_copy_states)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        p = GenerativeModel(config, tree)
        q = VarDistFixedTree(config, qc, qz, qeps, qmt, qpi, tree, y)
        copy_tree = CopyTree(config, p, q, y)

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
        C, y, z, pi, mu, tau, eps = self.simul_data_pyro_full_model(data, n_cells, n_sites, n_copy_states, tree,
                                                                    dir_alpha0=dir_alpha0)
        config = Config(n_nodes=K, chain_length=n_sites, n_cells=n_cells, n_states=n_copy_states)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        p = GenerativeModel(config, tree)
        q = VarDistFixedTree(config, qc, qz, qeps, qmt, qpi, tree, y)
        q.initialize(eps_alpha=10., eps_beta=40.,
                     loc=mu, precision_factor=.1, shape=5, rate=5)

        copy_tree = CopyTree(config, p, q, y)
        copy_tree.q.pi.concentration_param = dir_alpha0 * torch.ones(K)
        copy_tree.q.z.pi[...] = f.one_hot(z, num_classes=K)
        copy_tree.q.c.single_filtering_probs[...] = f.one_hot(C.long(), num_classes=n_copy_states).float()

        copy_tree.run(20)
        q_C = copy_tree.q.c.single_filtering_probs
        q_pi = copy_tree.q.z.pi
        delta = copy_tree.q.pi.concentration_param

        print(f"True dirichlet param: {dir_alpha0 * torch.ones(K)} \n variational concentration param: {delta}")
        print(f"True pi: {pi} \n variational concentration param: {torch.mean(q_pi, dim=0)}")
        print(f"True C: {f.one_hot(C[1, 5:10].long(), num_classes=n_copy_states)} \n q(C): {q_C[1, 5:10, :]}")
        print(f"True C: {f.one_hot(C[3, 5:10].long(), num_classes=n_copy_states)} \n q(C): {q_C[3, 5:10, :]}")
        # print(qmt.exp_tau())

        print(f'tree: {tree.edges}')
