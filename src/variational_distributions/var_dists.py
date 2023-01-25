import logging

import hmmlearn.hmm
import torch
import torch.nn.functional as torch_functional
import networkx as nx
import itertools
import numpy as np
from typing import List, Tuple, Union, Optional

from utils import math_utils
from utils.eps_utils import get_zipping_mask, get_zipping_mask0, h_eps

import utils.tree_utils as tree_utils
from sampling.slantis_arborescence import sample_arborescence, sample_arborescence_from_weighted_graph
from utils.config import Config
from variational_distributions.variational_distribution import VariationalDistribution


# copy numbers
class qC(VariationalDistribution):

    def __init__(self, config: Config, true_params=None):

        self._single_filtering_probs = torch.empty((config.n_nodes, config.chain_length, config.n_states))
        self._couple_filtering_probs = torch.empty(
            (config.n_nodes, config.chain_length - 1, config.n_states, config.n_states))

        # eta1 = log(pi) - log initial states probs
        self._eta1 = torch.empty((config.n_nodes, config.n_states))
        # eta2 = log(phi) - log transition probs
        self._eta2 = torch.empty_like(self._couple_filtering_probs)

        # validate true params
        if true_params is not None:
            assert "c" in true_params
        self.true_params = true_params
        super().__init__(config, fixed=true_params is not None)

    @property
    def single_filtering_probs(self):
        if self.fixed:
            cn_profile = self.true_params["c"]
            true_sfp = torch_functional.one_hot(cn_profile, num_classes=self.config.n_states).float()
            self._single_filtering_probs[...] = true_sfp
        return self._single_filtering_probs

    @single_filtering_probs.setter
    def single_filtering_probs(self, sfp):
        if self.fixed:
            logging.warning('Trying to re-set qc attribute when it should be fixed')
        self._single_filtering_probs[...] = sfp

    @property
    def couple_filtering_probs(self):
        if self.fixed:
            cn_profile = self.true_params["c"]
            # map node specific copy number profile to pair marginal probabilities
            # almost all prob mass is given to the true copy number combinations
            true_cfp = torch.zeros_like(self._couple_filtering_probs) + 1e-2 / (self.config.n_states ** 2 - 1)
            for u in range(self.config.n_nodes):
                true_cfp[u, torch.arange(self.config.chain_length - 1),
                         cn_profile[u, :-1], cn_profile[u, 1:]] = 1. - 1e-2
            self._couple_filtering_probs[...] = true_cfp
            if self.config.debug:
                assert torch.allclose(self._couple_filtering_probs.sum(dim=(2, 3)),
                                      torch.ones((self.config.n_nodes, self.config.chain_length-1)))

        return self._couple_filtering_probs

    @couple_filtering_probs.setter
    def couple_filtering_probs(self, cfp):
        if self.fixed:
            logging.warning('Trying to re-set qc attribute when it should be fixed')

        self._couple_filtering_probs[...] = cfp

    @property
    def eta1(self):
        return self._eta1

    @eta1.setter
    def eta1(self, e1):
        self._eta1[...] = e1

    @property
    def eta2(self):
        return self._eta2

    @eta2.setter
    def eta2(self, e2):
        self._eta2[...] = e2

    def initialize(self, **kwargs):
        if 'method' in kwargs.keys() and kwargs['method'] == 'baum-welch':
            self.baum_welch_init(obs=kwargs['obs'], qmt=kwargs['qmt'])
        else:
            self.random_init()
        return super().initialize(**kwargs)

    def random_init(self):
        self.eta1 = torch.rand(self.eta1.shape)
        self.eta1 = self.eta1 - torch.logsumexp(self.eta1, dim=-1, keepdim=True)
        self.eta2 = torch.rand(self.eta2.shape)
        self.eta2 = self.eta2 - torch.logsumexp(self.eta2, dim=-1, keepdim=True)

        self.compute_filtering_probs()

    def baum_welch_init(self, obs: torch.Tensor, qmt: 'qMuTau'):
        # distribute data to clones
        # calculate MLE state for each clone given the assigned cells
        A = self.config.n_states
        startprob_prior = torch.zeros(A)
        startprob_prior[2] = 1.
        c_tensor = torch.arange(A)
        mu_prior = qmt.nu.numpy()
        mu_prior_c = torch.outer(c_tensor, qmt.nu).numpy()
        hmm = hmmlearn.hmm.GaussianHMM(n_components=A,
                                       covariance_type='diag',
                                       means_prior=mu_prior_c,
                                       startprob_prior=startprob_prior,
                                       n_iter=100)
        hmm.fit(obs.numpy())
        self.eta1 = torch.log(torch.tensor(hmm.startprob_))
        self.eta2 = torch.log(torch.tensor(hmm.transmat_))
        self.eta2 = self.eta2 - torch.logsumexp(self.eta2, dim=-1, keepdim=True)

        self.compute_filtering_probs()

    def uniform_init(self):
        """
        Mainly used for testing.
        """
        self.eta1 = torch.ones(self.eta1.shape)
        self.eta1 = self.eta1 - torch.logsumexp(self.eta1, dim=-1, keepdim=True)
        self.eta2 = torch.ones(self.eta2.shape)
        self.eta2 = self.eta2 - torch.logsumexp(self.eta2, dim=-1, keepdim=True)

        self.compute_filtering_probs()

    def log_density(self, copy_numbers: torch.Tensor, nodes: list = []) -> float:
        # compute probability of a copy number sequence over a set of nodes
        # if the nodes are not specified, whole q_c is evaluated (all nodes)
        # copy_numbers has shape (nodes, chain_length)
        # TODO: it's more complicated than expected, fixing it later
        if len(nodes):
            assert copy_numbers.shape[0] == len(nodes)
            pass

        else:
            pass
        return 0.

    def entropy(self):
        qC_init = torch.distributions.Categorical(torch.exp(self.eta1))
        init_entropy = qC_init.entropy().sum()
        qC = torch.distributions.Categorical(torch.exp(self.eta2))
        transitions_entropy = qC.entropy().sum()
        #transitions_entropy = -torch.einsum("kmij, kmij ->", self.eta2, torch.log(self.eta2))
        return init_entropy + transitions_entropy

    def marginal_entropy(self):
        eps = 0.00001  # To avoid log_marginals having entries of -inf
        log_marginals = torch.log(self.single_filtering_probs + eps)
        return -torch.einsum("kmi, kmi ->", self.single_filtering_probs, log_marginals)

    def cross_entropy_old(self, T_list, w_T_list, q_eps: Union['qEpsilon', 'qEpsilonMulti']) -> float:
        # E_q[log p(C|...)]
        E_T = 0
        L = len(T_list)
        normalizing_weight = torch.logsumexp(torch.tensor(w_T_list), dim=0)
        for l in range(L):
            # FIXME: it's not exactly alpha1 and alpha2
            #   still, maybe it can be used by changing the nodes permutation
            alpha_1, alpha_2 = self._exp_alpha(T_list[l], q_eps)
            cross_ent_pos_1 = torch.einsum("ki,ki->",
                                           self.single_filtering_probs[:, 0, :],
                                           torch.log(alpha_1[:, :]))
            cross_ent_pos_2_to_M = torch.einsum("kmij,kmij->",
                                                self.couple_filtering_probs,
                                                alpha_2)
            E_T += torch.exp(w_T_list[l] - normalizing_weight) *\
                   (cross_ent_pos_1 + cross_ent_pos_2_to_M)

        return E_T

    def cross_entropy(self, T_list, w_T_list, q_eps: Union['qEpsilon', 'qEpsilonMulti']) -> float:
        # E_q[log p(C|...)]
        E_T = 0
        L = len(T_list)
        normalizing_weight = torch.logsumexp(torch.tensor(w_T_list), dim=0)
        for l in range(L):
            tree_CE = 0
            for a in T_list[l].edges:
                u, v = a
                arc_CE = self.cross_entropy_arc(q_eps, u, v)
                tree_CE += arc_CE

            E_T += torch.exp(w_T_list[l] - normalizing_weight) * tree_CE
        return E_T

    def cross_entropy_arc(self, q_eps, u, v):
        E_h_eps_0 = q_eps.h_eps0()
        E_h_eps = q_eps.exp_log_zipping((u, v))
        cross_ent_pos_1 = torch.einsum("i,j,ij->",
                                       self.single_filtering_probs[u, 0, :],
                                       self.single_filtering_probs[v, 0, :],
                                       E_h_eps_0)
        cross_ent_pos_2_to_M = torch.einsum("mik, mjl, ikjl->",
                                            self.couple_filtering_probs[u, :, :, :],
                                            self.couple_filtering_probs[v, :, :, :],
                                            E_h_eps)

        return cross_ent_pos_1 + cross_ent_pos_2_to_M

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
            exp_alpha1, exp_alpha2 = self._exp_alpha(tree, q_eps)

            # init tree-specific update
            tree_eta1, tree_eta2 = self._exp_eta(obs, tree, q_eps, q_z, q_mutau)
            for u in range(self.config.n_nodes):
                # for each node, get the children
                children = [w for w in tree.successors(u)]
                # sum on each node's update all children alphas
                alpha1sum = torch.einsum('wi->i', exp_alpha1[children, :])
                tree_eta1[u, :] += alpha1sum

                alpha2sum = torch.einsum('wmij->mij', exp_alpha2[children, :, :, :])
                tree_eta2[u, :, :, :] += alpha2sum

            new_eta1 = new_eta1 + tree_eta1 * weight
            new_eta2 = new_eta2 + tree_eta2 * weight

        # eta1 and eta2 don't come out normalized (in exp scale)
        # need normalization
        new_eta1_norm = new_eta1 - torch.logsumexp(new_eta1, dim=-1, keepdim=True)
        new_eta2_norm = new_eta2 - torch.logsumexp(new_eta2, dim=-1, keepdim=True)
        # update the filtering probs
        self.update_params(new_eta1_norm, new_eta2_norm)
        self.compute_filtering_probs()
        return super().update()

    def update_params(self, eta1, eta2):
        rho = self.config.step_size
        new_eta1 = (1 - rho) * torch.exp(self.eta1) + rho * torch.exp(eta1)
        new_eta2 = (1 - rho) * torch.exp(self.eta2) + rho * torch.exp(eta2)
        self.eta1 = torch.log(new_eta1)
        self.eta2 = torch.log(new_eta2)
        return new_eta1, new_eta2

    def _exp_eta(self, obs: torch.Tensor, tree: nx.DiGraph,
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

        e_eta1 = torch.empty_like(self.eta1)
        e_eta2 = torch.empty_like(self.eta2)

        # eta_1_iota(m, i)
        e_eta1_m = torch.einsum('nv,nmi->vmi',
                              q_z.exp_assignment(),
                              q_mutau.exp_log_emission(obs))
        # eta_1_iota(1, i)
        e_eta1[inner_nodes, :] = torch.einsum('pj,ij->pi',
                                              self.single_filtering_probs[
                                              [next(tree.predecessors(u)) for u in inner_nodes], 0, :],
                                              q_eps.h_eps0()) +\
                                e_eta1_m[inner_nodes, 0, :]
        # eta_2_kappa(m, i, i')
        if not isinstance(q_eps, qEpsilonMulti):
            e_eta2[...] = torch.einsum('pmjk,hikj->pmih',
                                       self.couple_filtering_probs,
                                       q_eps.exp_log_zipping()) +\
                                              e_eta1_m[:, 1:, None, :]
        else:
            edges_mask = [[p for p, _ in tree.edges], [v for _, v in tree.edges]]
            e_eta2[edges_mask[1], ...] = torch.einsum('pmjk,phikj->pmih',
                                                      self.couple_filtering_probs[edges_mask[0], ...],
                                                      torch.stack([q_eps.exp_log_zipping(e) for e in tree.edges])) +\
                                                          e_eta1_m[edges_mask[1], 1:, None, :]
            #e_eta2[edges_mask[1], ...] = torch.einsum('pmjk,phikj->pmih', #TODO: CHECK THIS
            #                                              self.couple_filtering_probs[edges_mask[0], ...],
            #                                              torch.stack([q_eps.exp_zipping(e) for e in tree.edges]))

        # natural parameters for root node are fixed to healthy state
        # FIXME: cells shouldn't be assigned to this node
        e_eta1[root, 2] = 0.  # exp(eta1_2) = pi_2 = 1.
        e_eta2[root, :, :, 2] = 0.  # exp(eta2_i2) = 1.

        all_but_2 = torch.arange(self.config.n_states) != 2
        e_eta1[root, all_but_2] = -torch.inf
        e_eta2[root, :, :, all_but_2] = -torch.inf

        return e_eta1, e_eta2

    def _exp_alpha(self, tree: nx.DiGraph, q_eps: Union['qEpsilon', 'qEpsilonMulti']) -> Tuple[
        torch.Tensor, torch.Tensor]:

        e_alpha1 = torch.empty_like(self.eta1)
        e_alpha2 = torch.empty_like(self.eta2)

        # alpha_iota(m, i)
        # as in the write-up, then it's split and e_alpha12[:, 1:, :] is
        # incorporated into e_alpha2
        e_alpha12 = torch.einsum('wmj,ji->wmi', self.single_filtering_probs, q_eps.h_eps0())

        e_alpha1[...] = e_alpha12[:, 0, :]  # first site goes to initial state

        # alpha_kappa(m, i, i')
        # similar to eta2 but with inverted indices in zipping
        if not isinstance(q_eps, qEpsilonMulti):
            e_alpha2 = torch.einsum('wmjk,kjhi->wmih',
                                    self.couple_filtering_probs,
                                    q_eps.exp_log_zipping((0, 0))) +\
                       e_alpha12[:, 1:, None, :]
        else:
            edges_mask = [[p for p, _ in tree.edges], [v for _, v in tree.edges]]
            e_alpha2[edges_mask[1], ...] = torch.einsum('wmjk,wkjhi->wmih',
                                                        self.couple_filtering_probs[edges_mask[1], ...],
                                                        torch.stack([q_eps.exp_log_zipping(e) for e in tree.edges])) +\
                                           e_alpha12[edges_mask[1], 1:, None, :]

        return e_alpha1, e_alpha2

    def mc_filter(self, u, m, i: Union[int, Tuple[int, int]]):
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

    def compute_filtering_probs(self):
        # shape K x S (K is batch size / clones)
        initial_log_probs = self.eta1
        # shape K x M x S x S
        transition_log_probs = self.eta2
        if self.config.debug:
            assert np.allclose(initial_log_probs.logsumexp(dim=1).exp(), 1.)
            assert np.allclose(transition_log_probs.logsumexp(dim=3).exp(), 1.)

        log_single = torch.empty_like(self.single_filtering_probs)
        log_couple = torch.empty_like(self.couple_filtering_probs)
        log_single[:, 0, :] = initial_log_probs
        for m in range(self.config.chain_length - 1):
            # first compute the two slice P(X_m, X_m+1) = P(X_m)P(X_m+1|X_m)
            log_couple[:, m, ...] = log_single[:, m, :, None] + transition_log_probs[:, m, ...]
            # then marginalize over X_m to obtain P(X_m+1)
            log_single[:, m + 1, :] = torch.logsumexp(log_couple[:, m, ...], dim=1)

        if self.config.debug:
            assert np.allclose(log_single.logsumexp(dim=2).exp(), 1.)
            assert np.allclose(log_couple.logsumexp(dim=(2, 3)).exp(), 1.)

        self.single_filtering_probs = torch.exp(log_single)
        self.couple_filtering_probs = torch.exp(log_couple)
        return self.single_filtering_probs, self.couple_filtering_probs

    def compute_fb_filtering_probs(self):
        # with forward-backward
        # TODO: compare with 'compute_filt_probs' and assess correctness
        self.single_filtering_probs = self.get_all_marginals()
        self.couple_filtering_probs = self.get_all_two_sliced_marginals()

    def get_two_slice_marginals(self, u):
        return tree_utils.two_slice_marginals_markov_chain(self.eta1[u], self.eta2[u])

    def get_marginals(self, u):
        return tree_utils.one_slice_marginals_markov_chain(self.eta1[u], self.eta2[u])

    def get_all_marginals(self):
        # TODO: optimize replacing for-loop with einsum operations
        q_C = torch.zeros(self.single_filtering_probs.shape)
        for u in range(self.config.n_nodes):
            init_eta = self.eta1[u, :]
            # FIXME: normalization shouldn't be necessary. check if it's already normlzd
            init_probs_qu = torch.exp(init_eta - torch.logsumexp(init_eta, dim=0))
            log_transition_probs = torch.exp(self.eta2[u])
            transition_probs = torch.exp(log_transition_probs -
                                         torch.logsumexp(log_transition_probs, dim=2, keepdim=True))
            if self.config.debug:
                assert torch.isclose(init_probs_qu.sum(), torch.tensor(1.0))
                M, A, A = transition_probs.shape
                for m in range(M):
                    tot_trans_prob = torch.sum(transition_probs[m], dim=1)
                    assert torch.allclose(tot_trans_prob, torch.ones(A))

            q_C[u, :, :] = tree_utils.one_slice_marginals_markov_chain(init_probs_qu, transition_probs)

        return q_C

    def get_all_two_sliced_marginals(self):
        # TODO: optimize replacing for-loop with einsum operations
        q_C_pairs = torch.zeros(self.couple_filtering_probs.shape)
        for u in range(self.config.n_nodes):
            init_eta = self.eta1[u, :]
            init_probs_qu = torch.exp(init_eta - torch.logsumexp(init_eta, dim=0))
            log_transition_probs = self.eta2[u]
            # FIXME: normalization shouldn't be necessary
            transition_probs = torch.exp(log_transition_probs -
                                         torch.logsumexp(log_transition_probs, dim=2, keepdim=True))
            q_C_pairs[u, :, :, :] = tree_utils.two_slice_marginals_markov_chain(init_probs_qu, transition_probs)

        return q_C_pairs


# cell assignments
class qZ(VariationalDistribution):
    def __init__(self, config: Config, true_params=None):
        self.pi = torch.empty((config.n_cells, config.n_nodes))

        if true_params is not None:
            assert "z" in true_params
        self.true_params = true_params
        super().__init__(config, true_params is not None)

    def initialize(self, method: str = 'random', **kwargs):
        if method == 'random':
            self._random_init()
        elif method == 'uniform':
            self._uniform_init()
        else:
            raise ValueError(f'method `{method}` for qZ initialization is not implemented')
        return super().initialize(**kwargs)

    def _random_init(self):
        # sample from a Dirichlet
        self.pi[...] = torch.distributions.Dirichlet(torch.ones_like(self.pi)).sample()

    def _uniform_init(self):
        # initialize to uniform probs among nodes
        self.pi[...] = torch.ones_like(self.pi) / self.config.n_nodes

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
        qc_kmj = qc.single_filtering_probs
        # expected log pi
        e_logpi = qpi.exp_log_pi()
        # Dnmj
        d_nmj = qmt.exp_log_emission(obs) # TODO: CHECK THIS *0.00001

        # op shapes: k + S_mS_j mkj nmj -> nk
        gamma = e_logpi + torch.einsum('kmj,nmj->nk', qc_kmj, d_nmj)
        # TODO: remove asserts
        assert gamma.shape == (self.config.n_cells, self.config.n_nodes)
        pi = torch.softmax(gamma, dim=1)
        assert self.pi.shape == (self.config.n_cells, self.config.n_nodes)
        new_pi = self.update_params(pi)
        return super().update()

    def update_params(self, pi: torch.Tensor):
        rho = self.config.step_size
        new_pi = (1 - rho) * self.pi + rho * pi
        self.pi[...] = new_pi
        return new_pi

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

    def __init__(self, config: Config, true_params=None):
        self._weighted_graph = nx.DiGraph()

        if true_params is not None:
            assert 'tree' in true_params
        self.true_params = true_params
        super().__init__(config, fixed=true_params is not None)

    @property
    def weighted_graph(self):
        return self._weighted_graph

    # TODO: implement with initialization instruction from the doc
    def initialize(self, **kwargs):
        # rooted graph with random weights in (0, 1)
        self._weighted_graph = self.init_fc_graph()
        return super().initialize(**kwargs)

    def cross_entropy(self):
        K = torch.tensor(self.config.n_nodes)
        return -torch.log(math_utils.cayleys_formula(K))

    def entropy(self):
        # TODO: product over edges in tree
        return 0

    def elbo(self) -> float:
        return self.cross_entropy() - self.entropy()

    def update(self, T_list, qc: qC, qeps: Union['qEpsilon', 'qEpsilonMulti']):
        q_T = self.update_CAVI(T_list, qc, qeps)
        self.update_graph_weights(qc, qeps)
        return q_T

    def update_graph_weights(self, qc: qC, qeps: Union['qEpsilon', 'qEpsilonMulti']):
        all_edges = [(u, v) for u, v in self._weighted_graph.edges]
        new_log_weights = {}
        for u, v in all_edges:
            # TODO: check that u, v order in einsum op is correct
            #   qt is updated but in the opposite way it should be
            new_log_weights[u, v] = torch.einsum('mij,mkl,jilk->', qc.couple_filtering_probs[u],
                                                 qc.couple_filtering_probs[v], qeps.exp_log_zipping((u, v)))
        w_tensor = torch.tensor(list(new_log_weights.values())).exp()
        # min-max scaling of weights
        w_tensor -= torch.min(w_tensor)
        w_tensor /= torch.max(w_tensor)
        for i, (u, v) in enumerate(new_log_weights):
            self._weighted_graph.edges[u, v]['weight'] = w_tensor[i]
        return super().update()

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
        for u, v in unique_edges:
            E_eps_h = q_epsilon.exp_log_zipping((u, v))
            E_CuCveps[u, v] = torch.einsum('mij, mkl, ijkl  -> ', q_C_pairwise_marginals[u], q_C_pairwise_marginals[v],
                                           E_eps_h)

        for k, T in enumerate(T_list):
            for u, v in T.edges:
                log_q_T_tensor[k] += E_CuCveps[u, v]

        return log_q_T_tensor

    def init_fc_graph(self, root=0):
        # random initialization of the fully connected graph over the clones
        g = nx.DiGraph()
        weighted_edges = [(u, v, torch.rand(1))
                          for u, v in itertools.permutations(range(self.config.n_nodes), 2)]
        g.add_weighted_edges_from(weighted_edges)
        # remove all edges going to the root
        edges_in_root = [(u, v) for u, v in g.in_edges(root)]
        g.remove_edges_from(edges_in_root)
        return g

    def get_trees_sample(self, alg="dslantis", sample_size=None) -> Tuple[List, List]:
        # e.g.:
        # trees = edmonds_tree_gen(self.config.is_sample_size)
        # trees = csmc_tree_gen(self.config.is_sample_size)
        l = sample_size
        trees = []
        log_weights = torch.empty(l)
        l = self.config.wis_sample_size if l is None else l
        if self.fixed:
            trees = [self.true_params['tree']] * l
            log_weights[...] = torch.ones(l)

        elif alg == "random":
            trees = [nx.random_tree(self.config.n_nodes, create_using=nx.DiGraph)
                     for _ in range(l)]
            log_weights[...] = torch.ones(l)
            for t in trees:
                nx.set_edge_attributes(t, np.random.rand(len(t.edges)), 'weight')

        elif alg == "dslantis":
            # nx.adjacency_matrix(self.weighted_graph, weight="weight") # doesn't work
            adj_matrix = nx.to_numpy_array(self.weighted_graph, weight="weight")
            log_W = torch.log(torch.tensor(adj_matrix))
            for i in range(l):
                # t, w = sample_arborescence(log_W=log_W, root=0)
                t, log_w = sample_arborescence_from_weighted_graph(self.weighted_graph)
                trees.append(t)
                log_weights[i] = log_w

        else:
            raise ValueError(f"alg '{alg}' is not implemented, check the documentation")

        # normalize weights and exponentiate
        log_weights[...] = log_weights - torch.logsumexp(log_weights, dim=0)
        weights = torch.exp(log_weights)
        if self.config.debug:
            assert torch.isclose(torch.sum(weights), torch.tensor(1.))
        return trees, weights.tolist()


# edge distance (single eps for all nodes)
class qEpsilon(VariationalDistribution):

    def __init__(self, config: Config, alpha_0: float = 1., beta_0: float = 1.):
        self.alpha_prior = torch.tensor(alpha_0, dtype=torch.float32)
        self.beta_prior = torch.tensor(beta_0, dtype=torch.float32)
        self.alpha = torch.tensor(alpha_0, dtype=torch.float32)
        self.beta = torch.tensor(beta_0, dtype=torch.float32)
        self._exp_log_zipping = None
        super().__init__(config)

    def update_params(self, alpha: torch.Tensor, beta: torch.Tensor):
        rho = self.config.step_size
        new_alpha = (1 - rho) * self.alpha + rho * alpha
        new_beta = (1 - rho) * self.beta + rho * beta
        self.alpha[...] = new_alpha
        self.beta[...] = new_beta
        self._exp_log_zipping = None  # reset previously computed expected zipping
        return new_alpha, new_beta

    def initialize(self, **kwargs):
        # TODO: implement (over set params)
        return super().initialize(**kwargs)

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
                                          anti_sym_mask)
            E_CuCv_b[u, v] = torch.einsum('mij, mkl, ijkl -> ', q_C_pairwise_marginals[u], q_C_pairwise_marginals[v],
                                          co_mut_mask)

        for k, T in enumerate(T_list):
            for uv in [e for e in T.edges]:
                u, v = uv
                alpha += w_T[k] * E_CuCv_a[u, v]
                beta += w_T[k] * E_CuCv_b[u, v]

        self.update_params(alpha, beta)
        return alpha, beta

    def h_eps0(self, i: Optional[int] = None, j: Optional[int] = None) -> Union[float, torch.Tensor]:
        if i is not None and j is not None:
            return 1. - self.config.eps0 if i == j else self.config.eps0 / (self.config.n_states - 1)
        else:
            heps0_arr = self.config.eps0 / (self.config.n_states - 1) * torch.ones((self.config.n_states, self.config.n_states))
            diag_mask = get_zipping_mask0(self.config.n_states)
            heps0_arr[diag_mask] = 1 - self.config.eps0
            if i is None and j is not None:
                return heps0_arr[:, j]
            elif i is not None and j is None:
                return heps0_arr[i, :]
            else:
                return heps0_arr

    def exp_log_zipping(self, _: Optional[Tuple[int, int]] = None):
        # indexing [j', j, i', i]
        if self._exp_log_zipping is None:
            self._exp_log_zipping = torch.empty_like((self.config.n_states, ) * 4)
            # bool tensor with True on [j', j, i', i] where j'-j = i'-i (comutation)
            comut_mask = get_zipping_mask(self.config.n_states)

            # exp( E_CuCv[ log( 1 - eps) ] )
            # switching to exponential leads to easier normalization step
            # (same as the one in `h_eps()`)
            exp_E_log_1meps = comut_mask * torch.exp(torch.digamma(self.beta) -
                                                     torch.digamma(self.alpha + self.beta))
            exp_E_log_eps = (1. - exp_E_log_1meps.sum(dim=0)) / torch.sum(~comut_mask, dim=0)
            self._exp_log_zipping[...] = exp_E_log_eps * (~comut_mask) + exp_E_log_1meps
            if self.config.debug:
                assert torch.allclose(torch.sum(self._exp_log_zipping, dim=0), torch.ones_like(self._exp_log_zipping))
            self._exp_log_zipping[...] = self._exp_log_zipping.log()

        return self._exp_log_zipping


# edge distance (multiple eps, one for each arc)
class qEpsilonMulti(VariationalDistribution):

    def __init__(self, config: Config, alpha_0: float = 1., beta_0: float = 1.,
                 true_params=None):
        # FIXME: alpha and beta should not be tensors, but rather dictionaries
        # so that only admitted arcs are present (and self arcs such as v->v are not accessible)
        self.alpha_prior = torch.tensor(alpha_0)
        self.beta_prior = torch.tensor(beta_0)
        # one param for every arc except self referenced (diag set to -infty)
        self._alpha = torch.diag(-torch.ones(config.n_nodes) * np.infty) + alpha_0
        self._beta = torch.diag(-torch.ones(config.n_nodes) * np.infty) + beta_0

        if true_params is not None:
            assert "eps" in true_params
        self.true_params = true_params
        super().__init__(config, true_params is not None)

    def update_params(self, alpha: torch.Tensor, beta: torch.Tensor):
        rho = self.config.step_size
        new_alpha = (1 - rho) * self.alpha + rho * alpha
        new_beta = (1 - rho) * self.beta + rho * beta
        self.alpha = new_alpha
        self.beta = new_beta
        return new_alpha, new_beta
    
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, a):
        self._alpha[...] = a

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, b):
        self._beta[...] = b

    def set_all_equal_params(self, alpha: float, beta: float):
        self.alpha = torch.diag(-torch.ones(self.config.n_nodes) * np.infty) + alpha
        self.beta = torch.diag(-torch.ones(self.config.n_nodes) * np.infty) + beta

    def initialize(self, method='uniform', **kwargs):
        if 'eps_alpha' in kwargs and 'eps_beta' in kwargs:
            self.set_all_equal_params(kwargs['eps_alpha'], kwargs['eps_beta'])
        elif method == 'uniform':
            self._uniform_init()
        elif method == 'random':
            self._random_init(**kwargs)
        else:
            raise ValueError(f'method `{method}` for qEpsilonMulti initialization is not implemented')
        return super().initialize(**kwargs)

    def _uniform_init(self):
        # results in uniform (0,1)
        self.alpha = torch.diag(-torch.ones(self.config.n_nodes) * np.infty) + 1.
        self.beta = torch.diag(-torch.ones(self.config.n_nodes) * np.infty) + 1.

    def _random_init(self, gamma_shape=2., gamma_rate=2.):
        a, b = torch.distributions.Gamma(gamma_shape, gamma_rate).sample(2)
        self.alpha = torch.diag(-torch.ones(self.config.n_nodes) * np.infty) + a
        self.beta = torch.diag(-torch.ones(self.config.n_nodes) * np.infty) + b

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
                                          anti_sym_mask)
            E_CuCv_b[u, v] = torch.einsum('mij, mkl, ijkl -> ',
                                          q_C_pairwise_marginals[u],
                                          q_C_pairwise_marginals[v],
                                          co_mut_mask)

        for k, T in enumerate(T_list):
            edges_mask = [[i for i, _ in T.edges], [j for _, j in T.edges]]
            # only change the values related to the tree edges
            alpha[edges_mask] += w_T[k] * E_CuCv_a[edges_mask]
            beta[edges_mask] += w_T[k] * E_CuCv_b[edges_mask]

        self.update_params(alpha, beta)
        return alpha, beta

    def h_eps0(self, i: Optional[int] = None, j: Optional[int] = None) -> Union[float, torch.Tensor]:
        if i is not None and j is not None:
            return 1. - self.config.eps0 if i == j else self.config.eps0 / (self.config.n_states - 1)
        else:
            heps0_arr = self.config.eps0 / (self.config.n_states - 1) * torch.ones((self.config.n_states, self.config.n_states))
            diag_mask = get_zipping_mask0(self.config.n_states)
            heps0_arr[diag_mask] = 1 - self.config.eps0
            if i is None and j is not None:
                return heps0_arr[:, j]
            elif i is not None and j is None:
                return heps0_arr[i, :]
            else:
                return heps0_arr

    def exp_log_zipping(self, e: Tuple[int, int]):
        """Expected log-zipping function

        Parameters
        ----------
        e : Tuple[int, int]
            edge associated to the distance epsilon
        output : indexing [j', j, i', i]
        """
        u, v = e
        out_arr = torch.empty((self.config.n_states, ) * 4)
        if self.fixed:
            # return the zipping function with true value of eps
            # which is the mean of the fixed distribution
            true_eps = self.true_params["eps"]
            try:
                out_arr[...] = torch.log(h_eps(self.config.n_states, true_eps[u, v]))
            except KeyError as ke:
                out_arr[...] = torch.log(h_eps(self.config.n_states, .8))  # distant clones if arc doesn't exist
        else:
            # bool tensor with True on [j', j, i', i] where j'-j = i'-i (comutation)
            comut_mask = get_zipping_mask(self.config.n_states)

            # exp( E_CuCv[ log( 1 - eps) ] )
            # switching to exponential leads to easier normalization step
            # (same as the one in `h_eps()`)
            exp_E_log_1meps = comut_mask * torch.exp(torch.digamma(self.beta[u, v]) -
                                                     torch.digamma(self.alpha[u, v] + self.beta[u, v]))
            exp_E_log_eps = (1. - exp_E_log_1meps.sum(dim=0)) / torch.sum(~comut_mask, dim=0)
            out_arr[...] = exp_E_log_eps * (~comut_mask) + exp_E_log_1meps
            if self.config.debug:
                assert torch.allclose(torch.sum(out_arr, dim=0), torch.ones_like(out_arr))
            out_arr[...] = out_arr.log()
        return out_arr

    def mean(self):
        return self.alpha / (self.alpha + self.beta)


# observations (mu-tau)
class qMuTau(VariationalDistribution):

    def __init__(self, config: Config, true_params=None):
        # params for each cell
        self._nu = torch.empty(config.n_cells)
        self._lmbda = torch.empty(config.n_cells)
        self._alpha = torch.empty(config.n_cells)
        self._beta = torch.empty(config.n_cells)
        self.nu_0 = torch.empty_like(self._nu)
        self.lmbda_0 = torch.empty_like(self._lmbda)
        self.alpha_0 = torch.empty_like(self._alpha)
        self.beta_0 = torch.empty_like(self._beta)

        if true_params is not None:
            # for each cell, mean and precision of the emission model
            assert "mu" in true_params
            assert "tau" in true_params
        self.true_params = true_params
        super().__init__(config, true_params is not None)

    # getter ensures that params are only updated in
    # the class' update method
    @property
    def nu(self):
        return self._nu

    @property
    def lmbda(self):
        return self._lmbda

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta
    
    @nu.setter
    def nu(self, n):
        self._nu[...] = n

    @lmbda.setter
    def lmbda(self, l):
        self._lmbda[...] = l

    @alpha.setter
    def alpha(self, a):
        self._alpha[...] = a

    @beta.setter
    def beta(self, b):
        self._beta[...] = b

    def update(self, qc: qC, qz: qZ, obs: torch.Tensor):
        """
        Updates mu_n, tau_n for each cell n \in {1,...,N}.
        :param qc:
        :param qz:
        :param obs: tensor of shape (M, N)
        :param sum_M_y2:
        :return:
        """
        A = self.config.n_states
        c_tensor = torch.arange(A, dtype=torch.float)
        q_Z = qz.exp_assignment()
        sum_MCZ_c2 = torch.einsum("kma, nk, a -> n", qc.single_filtering_probs, q_Z, c_tensor ** 2)
        sum_MCZ_cy = torch.einsum("kma, nk, a, mn -> n", qc.single_filtering_probs, q_Z, c_tensor, obs)
        sum_M_y2 = torch.pow(obs, 2).sum(dim=0)  # sum over M
        M = self.config.chain_length
        alpha = self.alpha_0 + M * .5  # Never updated
        lmbda = self.lmbda_0 + sum_MCZ_c2
        mu = (self.nu_0 * self.lmbda_0 + sum_MCZ_cy) / lmbda
        beta = self.beta_0 + .5 * (self.nu_0 ** 2 * self.lmbda_0 + sum_M_y2 - lmbda * mu ** 2)
        new_mu, new_lmbda, new_alpha, new_beta = self.update_params(mu, lmbda, alpha, beta)

        super().update()
        return new_mu, new_lmbda, new_alpha, new_beta

    def update_params(self, mu, lmbda, alpha, beta):
        rho = self.config.step_size
        new_nu = (1 - rho) * self._nu + rho * mu
        new_lmbda = (1 - rho) * self._lmbda + rho * lmbda
        new_alpha = (1 - rho) * self._alpha + rho * alpha
        new_beta = (1 - rho) * self._beta + rho * beta
        self.nu = new_nu
        self.lmbda = new_lmbda
        self.alpha = new_alpha
        self.beta = new_beta
        return new_nu, new_lmbda, new_alpha, new_beta

    def initialize(self, loc: float = 1, precision_factor: float = .1,
                 shape: float = 5, rate: float = 5, **kwargs):
        self.nu = loc * torch.ones(self.config.n_cells)
        self.lmbda = precision_factor * torch.ones(self.config.n_cells)
        self.alpha = shape * torch.ones(self.config.n_cells)
        self.beta = rate * torch.ones(self.config.n_cells)
        self.nu_0[...] = self._nu
        self.lmbda_0[...] = self._lmbda
        self.alpha_0[...] = self._alpha
        self.beta_0[...] = self._beta
        return super().initialize(**kwargs)

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
            E_log_tau = self.exp_log_tau()
            E_tau = torch.einsum('mn,n->mn', torch.pow(obs, 2), self.exp_tau())
            E_mu_tau = torch.einsum('i,mn,n->imn', torch.arange(self.config.n_states), obs, self.exp_mu_tau())
            E_mu2_tau = torch.einsum('i,n->in', torch.pow(torch.arange(self.config.n_states), 2), self.exp_mu2_tau())[:, None, :]
            out_arr = .5 * (E_log_tau - E_tau + 2.*E_mu_tau - E_mu2_tau)
            out_arr = torch.einsum('imn->nmi', out_arr)

        assert out_arr.shape == out_shape
        return out_arr

    def exp_tau(self):
        return self.alpha / self.beta

    def exp_log_tau(self):
        return torch.digamma(self.alpha) - torch.log(self.beta)

    def exp_mu_tau(self):
        return self.nu * self.alpha / self.beta

    def exp_mu2_tau(self):
        return 1. / self.lmbda + \
               torch.pow(self.nu, 2) * self.alpha / self.beta


# dirichlet concentration
class qPi(VariationalDistribution):

    def __init__(self, config: Config, alpha_prior=1, true_params=None):
        self.concentration_param_prior = torch.ones(config.n_nodes) * alpha_prior
        self._concentration_param = self.concentration_param_prior

        if true_params is not None:
            assert "pi" in true_params
        self.true_params = true_params
        super().__init__(config, fixed=true_params is not None)

    def update_params(self, concentration_param: torch.Tensor):
        rho = self.config.step_size
        new_concentration_param = (1 - rho) * self.concentration_param + rho * concentration_param
        self.concentration_param = new_concentration_param
        return new_concentration_param

    def initialize(self, method: str = 'random', **kwargs):
        if method == 'random':
            self._random_init()
        elif method == 'uniform':
            self._uniform_init()
        else:
            raise ValueError(f'method `{method}` for qZ initialization is not implemented')
        return super().initialize(**kwargs)

    def _uniform_init(self):
        # initialize to balanced concentration (all ones)
        self.concentration_param = torch.ones_like(self.concentration_param)

    def _random_init(self):
        self.concentration_param = torch.distributions.Gamma(5., 1.).rsample(self.concentration_param.shape)

    @property
    def concentration_param(self):
        return self._concentration_param
    
    @concentration_param.setter
    def concentration_param(self, cp):
        self._concentration_param[...] = cp
        
    def update(self, qz: qZ):
        # pi_model = p(pi), parametrized by delta_k
        # generative model for pi

        concentration_param = self.concentration_param_prior + \
                                   torch.sum(qz.exp_assignment(), dim=0)

        new_concentration_param = self.update_params(concentration_param)

        super().update()
        return new_concentration_param

    def exp_log_pi(self):
        e_log_pi = torch.empty_like(self.concentration_param)
        if self.fixed:
            e_log_pi[...] = torch.log(self.true_params["pi"])
        else:
            e_log_pi[...] = torch.digamma(self.concentration_param) - \
               torch.digamma(torch.sum(self.concentration_param))

        return e_log_pi

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
