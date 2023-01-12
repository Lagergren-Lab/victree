import logging

import torch
import torch.nn.functional as torch_functional
import networkx as nx
import itertools
import numpy as np
from typing import List, Tuple, Union, Optional

from utils import math_utils
from utils.eps_utils import get_zipping_mask, get_zipping_mask0, h_eps

import utils.tree_utils as tree_utils
from sampling.slantis_arborescence import sample_arborescence
from utils.config import Config
from variational_distributions.variational_distribution import VariationalDistribution


# copy numbers
class qC(VariationalDistribution):

    def __init__(self, config: Config):

        super().__init__(config)
        self.single_filtering_probs = torch.empty((self.config.n_nodes, self.config.chain_length, self.config.n_states))
        self.couple_filtering_probs = torch.empty(
            (self.config.n_nodes, self.config.chain_length - 1, self.config.n_states, self.config.n_states))

        self.eta1 = torch.empty((self.config.n_nodes, self.config.chain_length, self.config.n_states))
        self.eta2 = torch.empty(
            (self.config.n_nodes, self.config.chain_length - 1, self.config.n_states, self.config.n_states))

    def initialize(self):
        # TODO: implement initialization of parameters
        self.random_init()
        return super().initialize()

    def random_init(self):
        self.eta1 = torch.rand((self.config.n_nodes, self.config.chain_length, self.config.n_states))
        self.eta1 = torch.log(self.eta1 / torch.sum(self.eta1, dim=-1, keepdim=True))
        self.eta2 = torch.rand(
            (self.config.n_nodes, self.config.chain_length - 1, self.config.n_states, self.config.n_states))
        self.eta2 = torch.log(self.eta2 / torch.sum(self.eta2, dim=-1, keepdim=True))

        self.calculate_filtering_probs()

    def uniform_init(self):
        """
        Mainly used for testing.
        """
        self.eta1 = torch.ones((self.config.n_nodes, self.config.chain_length, self.config.n_states))
        self.eta1 = self.eta1 / torch.sum(self.eta1, dim=-1, keepdim=True)
        self.eta2 = torch.ones(
            (self.config.n_nodes, self.config.chain_length - 1, self.config.n_states, self.config.n_states))
        self.eta2 = self.eta2 / torch.sum(self.eta2, dim=-1, keepdim=True)

        self.calculate_filtering_probs()

    def log_density(self, copy_numbers: torch.Tensor, nodes: list = []) -> float:
        # compute probability of a copy number sequence over a set of nodes
        # if the nodes are not specified, whole q_c is evaluated (all nodes)
        # copy_numbers has shape (nodes, chain_length)
        # TODO: it's more complicated than expected, fixing it later
        if len(nodes):
            assert (copy_numbers.shape[0] == len(nodes))
            pass

        else:
            pass
        return 0.

    def entropy(self):
        qC_init = torch.distributions.Categorical(self.eta1)
        init_entropy = qC_init.entropy().sum()
        qC = torch.distributions.Categorical(self.eta2)
        transitions_entropy = qC.entropy().sum()
        #transitions_entropy = -torch.einsum("kmij, kmij ->", self.eta2, torch.log(self.eta2))
        return init_entropy + transitions_entropy

    def marginal_entropy(self):
        eps = 0.00001  # To avoid log_marginals having entries of -inf
        log_marginals = torch.log(self.single_filtering_probs + eps)
        return -torch.einsum("kmi, kmi ->", self.single_filtering_probs, log_marginals)

    def cross_entropy(self, T_list, w_T_list, q_eps: Union['qEpsilon', 'qEpsilonMulti']) -> float:
        E_T = 0
        L = len(T_list)
        normalising_weight = torch.logsumexp(torch.tensor(w_T_list), dim=0)
        for l in range(L):
            alpha_1, alpha_2 = self.exp_alpha(T_list[l], q_eps)
            cross_ent_pos_0 = torch.einsum("ki, ki -> ", self.single_filtering_probs[:, 0, :], torch.log(alpha_1[:, 0, :]))
            cross_ent_pos_2_to_M = torch.einsum("kmij, kmij -> ", self.couple_filtering_probs, alpha_2)
            E_T += torch.exp(w_T_list[l] - normalising_weight) * (cross_ent_pos_0 + cross_ent_pos_2_to_M)

        return E_T

    def elbo(self, T_list, w_T_list, q_eps: Union['qEpsilon', 'qEpsilonMulti']) -> float:
        # unique_arcs, unique_arcs_count = tree_utils.get_unique_edges(T_list, self.config.n_nodes)
        # for (a, a_count) in zip(unique_arcs, unique_arcs_count):
        # alpha_1, alpha_2 = self.exp_alpha()

        elbo = self.cross_entropy(T_list, w_T_list, q_eps) + self.marginal_entropy()
        return elbo

    def update(self, obs: torch.Tensor,
               q_eps: Union['qEpsilon', 'qEpsilonMulti'],
               q_z: 'qZ',
               q_mutau: 'qMuTau',
               trees,
               tree_weights) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        log q*(C) += ( E_q(mu)q(sigma)[rho_Y(Y^u, mu, sigma)] + E_q(T)[E_{C^p_u}[eta(C^p_u, epsilon)] +
        + Sum_{u,v in T} E_{C^v}[rho_C(C^v,epsilon)]] ) dot T(C^u)

        CAVI update based on the dot product of the sufficient statistic of the 
        HMM and simplified expected value over the natural parameter.
        :return:
        """
        new_eta1 = torch.zeros(self.eta1.shape)
        new_eta2 = torch.zeros(self.eta2.shape)

        for tree, weight in zip(trees, tree_weights):
            # compute all alpha quantities
            exp_alpha1, exp_alpha2 = self.exp_alpha(tree, q_eps)

            # init tree-specific update
            tree_eta1, tree_eta2 = self.exp_eta(obs, tree, q_eps, q_z, q_mutau)
            for u in range(self.config.n_nodes):
                # for each node, get the children
                children = [w for w in tree.successors(u)]
                # sum on each node's update all children alphas
                alpha1sum = torch.einsum('wmi->mi', exp_alpha1[children, :, :])
                tree_eta1[u, :, :] += alpha1sum
                
                alpha2sum = torch.einsum('wmij->mij', exp_alpha2[children, :, :, :])
                tree_eta2[u, :, :, :] += alpha2sum

            new_eta1 = new_eta1 + tree_eta1 * weight
            new_eta2 = new_eta2 + tree_eta2 * weight

        self.eta1 = new_eta1
        self.eta2 = new_eta2
        # update the filtering probs
        self.calculate_filtering_probs()
        return super().update()

    def exp_eta(self, obs: torch.Tensor, tree: nx.DiGraph,
                q_eps: Union['qEpsilon', 'qEpsilonMulti'],
                q_z: 'qZ',
                q_mutau: 'qMuTau') -> Tuple[torch.Tensor, torch.Tensor]:
        """Expectation of natural parameter vector \\eta

        Parameters
        ----------
        tree : nx.DiGraph
            Tree on which expectation is taken (e.g. sampled tree)
        q_eps : qEpsilon
            Variational distribution object of epsilon parameter
        q_z : VariationalDistribution
            Variational distribution object of cell assignment 
        q_mutau : qMuTau
            Variational distribution object of emission dist gaussian
            parameter (mu and tau)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor], concatenation of 3D array 
        eta1 and 4D array eta2 (expectation)
        """

        # get root node and make a mask 
        sorted_nodes = [u for u in nx.topological_sort(tree)]
        root = sorted_nodes[0]
        inner_nodes = sorted_nodes[1:]

        e_eta1 = torch.zeros(self.eta1.shape)
        e_eta2 = torch.zeros(self.eta2.shape)

        # eta_1_iota(1, i)
        e_eta1[inner_nodes, 0, :] = torch.einsum('pj,ij->pi',
                                                 self.single_filtering_probs[
                                                 [next(tree.predecessors(u)) for u in inner_nodes], 0, :],
                                                 q_eps.h_eps0())
        # eta_1_iota(m, i)
        e_eta1 = torch.einsum('nv,nmi->vmi',
                              q_z.exp_assignment(),
                              q_mutau.exp_log_emission(obs))
        # eta_2_kappa(m, i, i')
        if not isinstance(q_eps, qEpsilonMulti):
            e_eta2 = torch.einsum('pmjk,hikj->pmih',
                                              self.couple_filtering_probs,
                                              q_eps.exp_zipping())
        else:
            edges_mask = [[p for p, _ in tree.edges], [v for _, v in tree.edges]]
            e_eta2[edges_mask[1], ...] = torch.einsum('pmjk,phikj->pmih',
                                                          self.couple_filtering_probs[edges_mask[0], ...],
                                                          torch.stack([q_eps.exp_zipping(e) for e in tree.edges]))

        # natural parameters for root node are fixed to healthy state
        e_eta1[root, 0, 2] = 1
        e_eta2[root, :, :, 2] = 1

        return e_eta1, e_eta2

    def exp_alpha(self, tree: nx.DiGraph, q_eps: Union['qEpsilon', 'qEpsilonMulti']) -> Tuple[
        torch.Tensor, torch.Tensor]:

        e_alpha1 = torch.zeros(self.eta1.shape)
        e_alpha2 = torch.zeros(self.eta2.shape)

        # alpha_iota(m, i)
        e_alpha1 = torch.einsum('wmj,ji->wmi', self.single_filtering_probs, q_eps.h_eps0())

        # alpha_kappa(m, i, i')
        # similar to eta2 but with inverted indices in zipping
        if not isinstance(q_eps, qEpsilonMulti):
            e_alpha2 = torch.einsum('wmjk,kjhi->wmih',
                                                self.couple_filtering_probs,
                                                q_eps.exp_zipping((0, 0)))
        else:
            edges_mask = [[p for p, _ in tree.edges], [v for _, v in tree.edges]]
            e_alpha2[edges_mask[1], ...] = torch.einsum('wmjk,wkjhi->wmih',
                                                            self.couple_filtering_probs[edges_mask[1], ...],
                                                            torch.stack([q_eps.exp_zipping(e) for e in tree.edges]))

        return e_alpha1, e_alpha2

    def mc_filter(self, u, m, i: Union[int, Tuple[int, int]]):
        # TODO: implement/import forward-backward starting from eta params
        if isinstance(i, int):
            return self.single_filtering_probs[u, m, i]
        else:
            return self.couple_filtering_probs[u, m, i[0], i[1]]

    # iota/kappa
    # might be useless
    def idx_map(self, m, i: Union[int, Tuple[int, int]]) -> int:
        if isinstance(i, int):
            return m * self.config.n_states + i
        else:
            return m * (self.config.n_states ** 2) + i[0] * self.config.n_states + i[1]

    def calculate_filtering_probs(self):
        self.single_filtering_probs = self.get_all_marginals()
        self.couple_filtering_probs = self.get_all_two_sliced_marginals()

    def get_two_slice_marginals(self, u):
        return tree_utils.two_slice_marginals_markov_chain(self.eta1[u], self.eta2[u], self.config.chain_length)

    def get_marginals(self, u):
        return tree_utils.one_slice_marginals_markov_chain(self.eta1[u], self.eta2[u], self.config.chain_length)

    def get_all_marginals(self):
        # TODO: optimize replacing for-loop with einsum operations
        q_C = torch.zeros((self.config.n_nodes, self.config.chain_length, self.config.n_states))
        for u in range(self.config.n_nodes):
            init_eta = self.eta1[u, 0, :]
            init_probs_qu = torch.exp(init_eta - torch.logsumexp(init_eta, dim=0))
            transition_probs = torch.exp(self.eta2[u])
            transition_probs = torch_functional.normalize(transition_probs, p=1, dim=2)
            if self.config.debug:
                assert torch.isclose(init_probs_qu.sum(), torch.tensor(1.0))
                M, A, A = transition_probs.shape
                for m in range(M):
                    tot_trans_prob = torch.sum(transition_probs[m], dim=1)
                    assert torch.allclose(tot_trans_prob, torch.ones(A))

            q_C[u, :, :] = tree_utils.one_slice_marginals_markov_chain(init_probs_qu, transition_probs,
                                                                       self.config.chain_length)

        return q_C

    def get_all_two_sliced_marginals(self):
        # TODO: optimize replacing for-loop with einsum operations
        q_C_pairs = torch.zeros(self.couple_filtering_probs.shape)
        for u in range(self.config.n_nodes):
            init_eta = self.eta1[u, 0, :]
            init_probs_qu = torch.exp(init_eta - torch.logsumexp(init_eta, dim=0))
            transition_probs = torch.exp(self.eta2[u])
            transition_probs = torch_functional.normalize(transition_probs, p=1, dim=2)
            q_C_pairs[u, :, :, :] = tree_utils.two_slice_marginals_markov_chain(init_probs_qu, transition_probs,
                                                                                self.config.chain_length)

        return q_C_pairs


# cell assignments
class qZ(VariationalDistribution):
    def __init__(self, config: Config, true_params=None):
        self.pi = torch.empty((config.n_cells, config.n_nodes))
        self.true_params = true_params
        super().__init__(config, true_params is not None)

    def initialize(self):
        # initialize to uniform
        self.pi = torch.ones((self.config.n_cells, self.config.n_nodes)) / self.config.n_nodes
        return super().initialize()

    def update(self, qmt: 'qMuTau', qc: 'qC', qpi: 'qPi', obs: torch.Tensor):
        """
        q(Z) is a Categorical with probabilities pi^{*}, where pi_k^{*} = exp(gamma_k) / sum_K exp(gamma_k)
        gamma_k = E[log \pi_k] + sum_{m,j} q(C_m^k = j) E_{\mu, \tau}[D_{nmj}]
        :param Y: observations
        :param q_C_marginal:
        :param q_pi_dist:
        :param q_mu_tau:
        :return:
        """
        # single_filtering_probs: q(Cmk = j), shape: K x M x J
        qcmkj = qc.single_filtering_probs
        # expected log pi
        e_logpi = qpi.exp_log_pi()
        # Dnmj
        dnmj = qmt.exp_log_emission(obs)

        # op shapes: k + S_mS_j mkj nmj -> nk
        gamma = e_logpi + torch.einsum('kmj,nmj->nk', qcmkj, dnmj)
        # TODO: remove asserts
        assert (gamma.shape == (self.config.n_cells, self.config.n_nodes))
        self.pi = torch.softmax(gamma, dim=1)
        assert (self.pi.shape == (self.config.n_cells, self.config.n_nodes))

        return super().update()

    def exp_assignment(self) -> torch.Tensor:
        out_qz = torch.zeros((self.config.n_cells, self.config.n_nodes))
        if self.fixed:
            true_z = self.true_params["z"]
            # set prob of a true assignment to 1
            out_qz[torch.arange(self.config.n_cells), true_z] = 1.
        else:
            # simply the pi probabilities
            out_qz[...] = self.pi
        return out_qz

    def cross_entropy(self, qpi: 'qPi') -> float:
        e_logpi = qpi.exp_log_pi()
        return torch.einsum("nk, k -> ", self.pi, e_logpi)

    def entropy(self) -> float:
        return torch.special.entr(self.pi).sum()

    def elbo(self, qpi: 'qPi') -> float:
        return self.cross_entropy(qpi) + self.entropy()


# topology
class qT(VariationalDistribution):

    def __init__(self, config: Config):
        super().__init__(config)
        self.weighted_graph = self.init_fc_graph()

    # TODO: implement with initialization instruction from the doc
    def initialize(self):
        return super().initialize()

    def cross_entropy(self):
        K = torch.tensor(self.config.n_nodes)
        return -torch.log(math_utils.cayleys_formula(K))

    def entropy(self):
        # TODO: product over edges in tree
        return 0

    def elbo(self) -> float:
        return self.cross_entropy() - self.entropy()

    def update(self, T_list, q_C: qC, q_epsilon: Union['qEpsilon', 'qEpsilonMulti']):
        q_T = self.update_CAVI(T_list, q_C, q_epsilon)
        return q_T

    def update_CAVI(self, T_list: list, q_C: qC, q_epsilon: Union['qEpsilon', 'qEpsilonMulti']):
        """
        log q(T) =
        (1) E_{C^r}[log p(C^r)] +
        (2) sum_{uv in A(T)} E_{C^u, C^v, epsilon}[log p(C^u | C^v, epsilon)] +
        (3) log p(T)
        :param T_list: list of tree topologies (L x 1)
        :param q_C_pairwise_marginals: (N x M-1 x A x A)
        :param q_C: q(C) variational distribution object
        :param q_epsilon: q(epsilon) variational distribution object
        :return: log_q_T_tensor
        """
        K = len(T_list)
        q_C_pairwise_marginals = q_C.couple_filtering_probs
        N, M, A, A = q_C_pairwise_marginals.size()
        log_q_T_tensor = torch.zeros((K,))
        unique_edges, unique_edges_count = tree_utils.get_unique_edges(T_list, N)

        # Term (1) - expectation over root node
        # Constant w.r.t. T, can be omitted
        # Term (3)
        # Constant w.r.t T, can be omitted

        # Term (2)

        E_CuCveps = torch.zeros((N, N))
        for (u, v) in unique_edges:
            E_eps_h = q_epsilon.exp_zipping((u, v))
            E_CuCveps[u, v] = torch.einsum('mij, mkl, ijkl  -> ', q_C_pairwise_marginals[u], q_C_pairwise_marginals[v],
                                           E_eps_h)

        for (k, T) in enumerate(T_list):
            for (u, v) in T.edges:
                log_q_T_tensor[k] += E_CuCveps[u, v]

        return log_q_T_tensor

    def init_fc_graph(self):
        # random initialization of the fully connected graph over the clones
        g = nx.DiGraph()
        weighted_edges = [(u, v, torch.rand(1))
                          for u, v in itertools.permutations(range(self.config.n_nodes), 2)]
        g.add_weighted_edges_from(weighted_edges)
        return g

    def get_trees_sample(self, alg="dslantis", L=None) -> Tuple[List, List]:
        # TODO: generate trees with sampling algorithm
        # e.g.:
        # trees = edmonds_tree_gen(self.config.is_sample_size)
        # trees = csmc_tree_gen(self.config.is_sample_size)
        trees = []
        weights = []
        L = self.config.wis_sample_size if L is None else L
        if alg == "random":
            trees = [nx.random_tree(self.config.n_nodes, create_using=nx.DiGraph)
                     for _ in range(L)]
            weights = [1] * L
            for t in trees:
                nx.set_edge_attributes(t, np.random.rand(len(t.edges)), 'weight')

        elif alg == "dslantis":
            # nx.adjacency_matrix(self.weighted_graph, weight="weight") # doesn't work
            adj_matrix = nx.to_numpy_array(self.weighted_graph, weight="weight")
            log_W = torch.log(torch.tensor(adj_matrix))
            for _ in range(L):
                t, w = sample_arborescence(log_W=log_W, root=0)
                trees.append(t)
                weights.append(w)

        else:
            raise ValueError(f"alg '{alg}' is not implemented, check the documentation")

        return trees, weights


# edge distance (single eps for all nodes)
class qEpsilon(VariationalDistribution):

    def __init__(self, config: Config, alpha_0: float = 1., beta_0: float = 1.):
        self.alpha_prior = torch.tensor(alpha_0, dtype=torch.float32)
        self.beta_prior = torch.tensor(beta_0, dtype=torch.float32)
        self.alpha = torch.tensor(alpha_0, dtype=torch.float32)
        self.beta = torch.tensor(beta_0, dtype=torch.float32)
        self._exp_zipping = None
        super().__init__(config)

    def set_params(self, alpha: torch.Tensor, beta: torch.Tensor):
        self.alpha = alpha
        self.beta = beta
        self._exp_zipping = None  # reset previously computed expected zipping

    def initialize(self):
        # TODO: implement (over set params)
        return super().initialize()

    def create_masks(self, A):
        co_mut_mask = torch.zeros((A, A, A, A))
        anti_sym_mask = torch.zeros((A, A, A, A))
        # TODO: Find effecient way of indexing i-j = k-l
        for i, j, k, l in itertools.combinations_with_replacement(range(A), 4):
            if i - j == k - l:
                co_mut_mask[i, j, k, l] = 1
            else:
                anti_sym_mask[i, j, k, l] = 1
        return co_mut_mask, anti_sym_mask

    def elbo(self) -> float:
        return super().elbo()

    def update(self, T_list, w_T, q_C_pairwise_marginals):
        self.update_CAVI(T_list, w_T, q_C_pairwise_marginals)
        super().update()

    def update_CAVI(self, T_list: list, w_T: torch.Tensor, q_C_pairwise_marginals: torch.Tensor):
        N, M, A, A = q_C_pairwise_marginals.size()
        alpha = self.alpha_prior
        beta = self.beta_prior
        unique_edges, unique_edges_count = tree_utils.get_unique_edges(T_list, N_nodes=N)

        E_CuCv_a = torch.zeros((N, N))
        E_CuCv_b = torch.zeros((N, N))
        co_mut_mask, anti_sym_mask = self.create_masks(A)
        for uv in unique_edges:
            u, v = uv
            E_CuCv_a[u, v] = torch.einsum('mij, mkl, ijkl -> ', q_C_pairwise_marginals[u], q_C_pairwise_marginals[v],
                                          co_mut_mask)
            E_CuCv_b[u, v] = torch.einsum('mij, mkl, ijkl -> ', q_C_pairwise_marginals[u], q_C_pairwise_marginals[v],
                                          anti_sym_mask)

        for k, T in enumerate(T_list):
            for uv in [e for e in T.edges]:
                u, v = uv
                alpha += w_T[k] * E_CuCv_a[u, v]
                beta += w_T[k] * E_CuCv_b[u, v]

        self.set_params(alpha, beta)
        return alpha, beta

    def h_eps0(self, i: Optional[int] = None, j: Optional[int] = None) -> Union[float, torch.Tensor]:
        # FIXME: add normalization constant A0
        if i is not None and j is not None:
            return 1. - self.config.eps0 if i == j else self.config.eps0
        else:
            heps0_arr = self.config.eps0 * torch.ones((self.config.n_states, self.config.n_states))
            diag_mask = get_zipping_mask0(self.config.n_states)
            heps0_arr[diag_mask] = 1 - self.config.eps0
            if i is None and j is not None:
                return heps0_arr[:, j]
            elif i is not None and j is None:
                return heps0_arr[i, :]
            else:
                return heps0_arr

    def exp_zipping(self, _: Optional[Tuple[int, int]] = None):
        if self._exp_zipping is None:
            # TODO: implement
            copy_mask = get_zipping_mask(self.config.n_states)

            # FIXME: add normalization (division by A constant)
            norm_const = self.config.n_states

            out_arr = torch.ones(copy_mask.shape) * \
                      (torch.digamma(self.beta) - \
                       torch.digamma(self.alpha + self.beta))
            out_arr[~copy_mask] -= norm_const
            self._exp_zipping = out_arr

        return self._exp_zipping


# edge distance (multiple eps, one for each arc)
class qEpsilonMulti(VariationalDistribution):

    def __init__(self, config: Config, alpha_0: float = 1., beta_0: float = 1.,
                 true_params=None):
        self.alpha_prior = torch.tensor(alpha_0)
        self.beta_prior = torch.tensor(beta_0)
        # one param for every arc except self referenced (diag set to -infty)
        self.alpha = torch.diag(-torch.ones(config.n_nodes) * np.infty) + alpha_0
        self.beta = torch.diag(-torch.ones(config.n_nodes) * np.infty) + beta_0
        self.true_params = true_params

        super().__init__(config, true_params is not None)

    def set_params(self, alpha: torch.Tensor, beta: torch.Tensor):
        self.alpha = alpha
        self.beta = beta

    def initialize(self):
        # TODO: implement (over set params)
        return super().initialize()

    def create_masks(self, A):
        co_mut_mask = torch.zeros((A, A, A, A))
        anti_sym_mask = torch.zeros((A, A, A, A))
        # TODO: Find effecient way of indexing i-j = k-l
        for i, j, k, l in itertools.combinations_with_replacement(range(A), 4):
            if i - j == k - l:
                co_mut_mask[i, j, k, l] = 1
            else:
                anti_sym_mask[i, j, k, l] = 1
        return co_mut_mask, anti_sym_mask

    def cross_entropy(self):
        return 0

    def entropy(self):
        return 0

    def elbo(self) -> float:
        self.entropy()
        return super().elbo()

    def update(self, T_list, w_T, q_C_pairwise_marginals):
        self.update_CAVI(T_list, w_T, q_C_pairwise_marginals)
        super().update()

    def update_CAVI(self, T_list: list, w_T: torch.Tensor, q_C_pairwise_marginals: torch.Tensor):
        K, M, A, A = q_C_pairwise_marginals.shape
        alpha = torch.zeros(self.alpha.shape) + self.alpha_prior
        beta = torch.zeros(self.beta.shape) + self.beta_prior
        unique_edges, unique_edges_count = tree_utils.get_unique_edges(T_list, N_nodes=K)

        E_CuCv_a = torch.zeros((K, K))
        E_CuCv_b = torch.zeros((K, K))
        co_mut_mask, anti_sym_mask = self.create_masks(A)
        for uv in unique_edges:
            u, v = uv
            E_CuCv_a[u, v] = torch.einsum('mij, mkl, ijkl -> ',
                                          q_C_pairwise_marginals[u],
                                          q_C_pairwise_marginals[v],
                                          co_mut_mask)
            E_CuCv_b[u, v] = torch.einsum('mij, mkl, ijkl -> ',
                                          q_C_pairwise_marginals[u],
                                          q_C_pairwise_marginals[v],
                                          anti_sym_mask)

        for k, T in enumerate(T_list):
            edges_mask = [[i for i, _ in T.edges], [j for _, j in T.edges]]
            # only change the values related to the tree edges
            alpha[edges_mask] += w_T[k] * E_CuCv_a[edges_mask]
            beta[edges_mask] += w_T[k] * E_CuCv_b[edges_mask]

        self.set_params(alpha, beta)
        return alpha, beta

    def h_eps0(self, i: Optional[int] = None, j: Optional[int] = None) -> Union[float, torch.Tensor]:
        # FIXME: add normalization constant A0
        if i is not None and j is not None:
            return 1. - self.config.eps0 if i == j else self.config.eps0
        else:
            heps0_arr = self.config.eps0 * torch.ones((self.config.n_states, self.config.n_states))
            diag_mask = get_zipping_mask0(self.config.n_states)
            heps0_arr[diag_mask] = 1 - self.config.eps0
            if i is None and j is not None:
                return heps0_arr[:, j]
            elif i is not None and j is None:
                return heps0_arr[i, :]
            else:
                return heps0_arr

    def exp_zipping(self, e: Tuple[int, int]):
        """Expected zipping function

        Parameters
        ----------
        e : Tuple[int, int]
            edge associated to the distance epsilon
        """
        u, v = e
        if self.fixed:
            # return the zipping function with true value of eps
            # which is the mean of the fixed distribution
            true_eps = self.true_params["eps"]
            out_arr = h_eps(self.config.n_states, true_eps[u, v])
        else:
            # TODO: implement
            copy_mask = get_zipping_mask(self.config.n_states)

            # FIXME: add normalization (division by A constant)
            norm_const = self.config.n_states

            out_arr = torch.ones(copy_mask.shape) * \
                      (torch.digamma(self.beta[u, v]) - \
                       torch.digamma(self.alpha[u, v] + self.beta[u, v]))
            # select the combinations that do not satisfy i-i'=j-j'
            # and normalize
            out_arr[~copy_mask] -= norm_const
        return out_arr


# observations (mu-tau)
class qMuTau(VariationalDistribution):

    def __init__(self, config: Config, loc: float = 100, precision_factor: float = .1,
                 shape: float = 5, rate: float = 5, true_params=None):
        # params for each cell
        self._loc = loc * torch.ones(config.n_cells)
        self._precision_factor = precision_factor * torch.ones(config.n_cells)
        self._shape = shape * torch.ones(config.n_cells)
        self._rate = rate * torch.ones(config.n_cells)
        self.mu_prior = self._loc
        self.lambda_prior = self._precision_factor
        self.alpha_prior = self._shape
        self.alpha = self.alpha_prior + config.chain_length / 2  # alpha never updated
        self.beta_prior = self._rate
        self.true_params = true_params

        if true_params is not None:
            # for each cell, mean and precision of the emission model
            assert("mu" in true_params)
            assert("tau" in true_params)

        super().__init__(config, true_params is not None)

    # getter ensures that params are only updated in
    # the class' update method
    @property
    def loc(self):
        return self._loc

    @property
    def precision(self):
        return self._precision_factor

    @property
    def shape(self):
        return self._shape

    @property
    def rate(self):
        return self._rate

    def update(self, qc: qC, qz: qZ, obs, sum_M_y2):
        """
        Updates mu_n, tau_n for each cell n \in {1,...,N}.
        :param qc:
        :param qz:
        :param obs:
        :param sum_M_y2:
        :return:
        """
        A = self.config.n_states
        c_tensor = torch.arange(A, dtype=torch.float)
        sum_MCZ_c2 = torch.einsum("kma, nk, a -> n", qc.single_filtering_probs, qz.pi, c_tensor ** 2)
        sum_MCZ_cy = torch.einsum("kma, nk, a, mn -> n", qc.single_filtering_probs, qz.pi, c_tensor, obs)
        M = self.config.chain_length
        alpha = self.alpha_prior + M / 2  # Never updated
        lmbda = self.lambda_prior + sum_MCZ_c2
        mu = (self.mu_prior * self.lambda_prior + sum_MCZ_cy) / lmbda
        beta = self.beta_prior + 1 / 2 * (self.mu_prior ** 2 * self.lambda_prior + sum_M_y2) + \
               (self.mu_prior * self.lambda_prior + sum_MCZ_cy) ** 2 / (2 * lmbda)
        self._loc = mu
        self._precision_factor = lmbda
        self._shape = alpha
        self._rate = beta

        super().update()
        return mu, lmbda, alpha, beta

    def initialize(self):
        return super().initialize()

    def cross_entropy(self) -> float:
        return super().elbo()

    def entropy(self) -> float:
        return super().elbo()

    def elbo(self) -> float:
        return self.cross_entropy() + self.entropy()

    def exp_log_emission(self, obs: torch.Tensor) -> torch.Tensor:
        out_shape = (self.config.n_cells, self.config.chain_length, self.config.n_states)
        out_arr = torch.ones(out_shape)
        # obs is (m x n)
        if self.fixed:
            mu = self.true_params["mu"]
            tau = self.true_params["tau"]

            # log emission is log normal with
            # mean=mu*cn_state, var=1/tau
            means = torch.outer(mu,
                        torch.arange(self.config.n_states))
            true_dist = torch.distributions.Normal(loc=means,
                                       scale=torch.ones(means.shape) / torch.sqrt(tau)[:, None])
            out_arr = torch.permute(true_dist.log_prob(obs[..., None]), (1, 0, 2))
        else:
            out_arr = .5 * self.exp_log_tau() - \
                      .5 * torch.einsum('mn,n->mn',
                                        obs, self.exp_tau()) + \
                      torch.einsum('i,mn,n->imn',
                                   torch.arange(self.config.n_states),
                                   obs,
                                   self.exp_mu_tau()) - \
                      .5 * torch.einsum('i,n->in',
                                        torch.pow(torch.arange(self.config.n_states), 2),
                                        self.exp_mu2_tau())[:, None, :]
            out_arr = torch.einsum('imn->nmi', out_arr)

        assert out_arr.shape == out_shape
        return out_arr

    def exp_tau(self):
        return self.shape / self.rate

    def exp_log_tau(self):
        return torch.digamma(self.shape) - torch.log(self.rate)

    def exp_mu_tau(self):
        return self.loc * self.shape / self.rate

    def exp_mu2_tau(self):
        return 1. / self.precision + \
               torch.pow(self.loc, 2) * self.shape / self.rate


# dirichlet concentration
class qPi(VariationalDistribution):

    def __init__(self, config: Config, alpha_prior=1):
        super().__init__(config)
        self.concentration_param_prior = torch.ones(config.n_nodes) * alpha_prior
        self.concentration_param = self.concentration_param_prior

    def initialize(self):
        # initialize to balanced concentration (all ones)
        self.concentration_param = torch.ones(self.concentration_param.shape)
        return super().initialize()

    def update(self, qz: qZ):
        # pi_model = p(pi), parametrized by delta_k
        # generative model for pi

        self.concentration_param = self.concentration_param_prior + \
                                   torch.sum(qz.exp_assignment(), dim=0)

        return super().update()

    def exp_log_pi(self):
        return torch.digamma(self.concentration_param) - \
               torch.digamma(torch.sum(self.concentration_param))

    def cross_entropy(self):
        delta_p = self.concentration_param_prior
        delta_q = self.concentration_param
        delta_p_0 = torch.sum(delta_p)
        delta_q_0 = torch.sum(delta_q)
        K = delta_p.shape[0]
        digamma_q = torch.digamma(delta_q)
        digamma_q_0 = torch.digamma(delta_q_0)
        log_B_p = math_utils.log_beta_function(delta_p)
        return log_B_p + (K - delta_p_0) * digamma_q_0 + torch.sum((delta_p - 1) * digamma_q)

    def entropy(self):
        delta_q = self.concentration_param
        delta_q_0 = torch.sum(delta_q)
        K = delta_q.shape[0]
        digamma_q = torch.digamma(delta_q)
        digamma_q_0 = torch.digamma(delta_q_0)
        log_B_q = math_utils.log_beta_function(delta_q)
        return log_B_q + (delta_q_0 - K) * digamma_q_0 - torch.sum((delta_q - 1) * digamma_q)

    def elbo(self) -> float:
        cross_entropy = self.cross_entropy()
        entropy = self.entropy()
        return cross_entropy + entropy
