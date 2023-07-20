"""
Variational distribution classes with several initialization methods, update formulas and partial ELBO computation.
"""
import copy
import logging
import os

import hmmlearn.hmm
import torch
import torch.nn.functional as torch_functional
import networkx as nx
import itertools
import numpy as np
from typing import List, Tuple, Union, Optional

from utils.data_handling import dict_to_tensor, edge_dict_to_matrix
from utils.evaluation import pm_uni

from sklearn.cluster import KMeans

from utils import math_utils
from utils.eps_utils import get_zipping_mask, get_zipping_mask0, h_eps, normalizing_zipping_constant, \
    get_zipping_mask_old

import utils.tree_utils as tree_utils
from sampling.laris import sample_arborescence_from_weighted_graph
from utils.config import Config
from variational_distributions.observational_variational_distribution import qPsi
from variational_distributions.variational_distribution import VariationalDistribution


# ---
# Copy numbers
# ---
class qC(VariationalDistribution):

    def __init__(self, config: Config, true_params=None):
        """
        Variational distribution for copy number profiles, i.e. Markov chains for each cluster of cells.
        Parameters
        ----------
        config: Config, configuration object
        true_params: dict, contains "c" key which is a torch.Tensor of shape (n_nodes, chain_length) with
            copy number integer values
        """
        super().__init__(config, fixed=true_params is not None)

        self._single_filtering_probs = torch.empty((config.n_nodes, config.chain_length, config.n_states))
        self._couple_filtering_probs = torch.empty(
            (config.n_nodes, config.chain_length - 1, config.n_states, config.n_states))

        # eta1 = log(pi) - log initial states probs
        self._eta1: torch.Tensor = torch.empty((config.n_nodes, config.n_states))
        # eta2 = log(phi) - log transition probs
        self._eta2: torch.Tensor = torch.empty_like(self._couple_filtering_probs)

        # validate true params
        if true_params is not None:
            assert "c" in true_params
        self.true_params = true_params

        # define dist param names
        self.params_history["single_filtering_probs"] = []
        # # not needed at the moment
        # self.params_history["couple_filtering_probs"] = []

    @property
    def single_filtering_probs(self):
        if self.fixed:
            small_eps = 1e-5
            cn_profile = self.true_params["c"]
            true_sfp = torch_functional.one_hot(cn_profile,
                                                num_classes=self.config.n_states).float().clamp(min=small_eps)
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
            small_eps = 1e-5
            cn_profile = self.true_params["c"]
            # map node specific copy number profile to pair marginal probabilities
            # almost all prob mass is given to the true copy number combinations
            true_cfp = torch.zeros_like(self._couple_filtering_probs)
            for u in range(self.config.n_nodes):
                true_cfp[u, torch.arange(self.config.chain_length - 1),
                         cn_profile[u, :-1], cn_profile[u, 1:]] = 1.
            # add epsilon and normalize
            self._couple_filtering_probs[...] = true_cfp.clamp(min=small_eps)
            self._couple_filtering_probs /= self._couple_filtering_probs.sum(dim=(2, 3), keepdim=True)
            if self.config.debug:
                assert torch.allclose(self._couple_filtering_probs.sum(dim=(2, 3)),
                                      torch.ones((self.config.n_nodes, self.config.chain_length - 1)))

        return self._couple_filtering_probs

    @couple_filtering_probs.setter
    def couple_filtering_probs(self, cfp):
        if self.fixed:
            logging.warning('Trying to re-set qc attribute when it should be fixed')

        self._couple_filtering_probs[...] = cfp

    @property
    def eta1(self):
        if self.fixed:
            self._eta1[...] = self._single_filtering_probs[:, 0, :].log()
        return self._eta1

    @eta1.setter
    def eta1(self, e1):
        if self.fixed:
            logging.warning('Trying to re-set qc attribute when it should be fixed')
        self._eta1[...] = e1

    @property
    def eta2(self):
        if self.fixed:
            self._eta2[...] = self._couple_filtering_probs.log()
        return self._eta2

    @eta2.setter
    def eta2(self, e2):
        if self.fixed:
            logging.warning('Trying to re-set qc attribute when it should be fixed')
        self._eta2[...] = e2

    def initialize(self, method='random', **kwargs):
        if method == 'baum-welch':
            self._baum_welch_init(**kwargs)
        elif method == 'bw-cluster':
            self._init_bw_cluster_data(**kwargs)
        elif method == 'random':
            self._random_init()
        elif method == 'uniform':
            self._uniform_init()
        else:
            raise ValueError(f'method `{method}` for qC initialization is not implemented')

        return super().initialize(**kwargs)

    def _random_init(self):
        self.eta1 = torch.rand(self.eta1.shape)
        self.eta1 = self.eta1 - torch.logsumexp(self.eta1, dim=-1, keepdim=True)
        self.eta2 = torch.rand(self.eta2.shape)
        self.eta2 = self.eta2 - torch.logsumexp(self.eta2, dim=-1, keepdim=True)

        self.compute_filtering_probs()

    def _baum_welch_init(self, obs: torch.Tensor, qmt: 'qMuTau'):
        # TODO: test
        # distribute data to clones
        # calculate MLE state for each clone given the assigned cells
        A = self.config.n_states
        eps = .01
        startprob_prior = torch.zeros(A) + eps
        startprob_prior[2] = 1.
        startprob_prior = startprob_prior / torch.sum(startprob_prior)
        c_tensor = torch.arange(A)
        mu_prior = qmt.nu.numpy()
        mu_prior_c = torch.outer(c_tensor, qmt.nu).numpy()
        hmm = hmmlearn.hmm.GaussianHMM(n_components=A,
                                       covariance_type='diag',
                                       means_prior=mu_prior_c,
                                       startprob_prior=startprob_prior,
                                       n_iter=100)
        hmm_2 = hmmlearn.hmm.GMMHMM(n_components=A,
                                    covariance_type='diag',
                                    means_prior=mu_prior_c,
                                    startprob_prior=startprob_prior,
                                    n_iter=100)
        hmm.fit(obs.numpy())
        self.eta1 = torch.log(torch.tensor(hmm.startprob_))
        self.eta2 = torch.log(torch.tensor(hmm.transmat_))
        self.eta2 = self.eta2 - torch.logsumexp(self.eta2, dim=-1, keepdim=True)

        self.compute_filtering_probs()

    def _init_bw_cluster_data(self, obs: torch.Tensor, clusters):
        # qz must be initialized already
        zero_eps = 1e-5
        means = torch.empty(self.config.n_cells)
        precisions = torch.empty(self.config.n_cells)
        # distribute data to clones
        # skip root (node 0)
        for k in range(1, self.config.n_nodes):
            # obs is shape (n_samples, n_features)
            # where n_samples is simply the chain length (one observation)
            # and n_features=n_cells, in order to get mu/precision for each cell
            # cov_type is diag, so to consider independent cells

            hmm = hmmlearn.hmm.GaussianHMM(n_components=self.config.n_states,
                                           implementation='log',
                                           covariance_type='diag',
                                           n_iter=100)

            hmm.fit(obs[:, clusters == k], lengths=[self.config.chain_length])

            self.eta1[k, :] = torch.tensor(hmm.startprob_).clamp(min=zero_eps).log()
            self.eta1[k, :] = self.eta1[k, :] - torch.logsumexp(self.eta1[k, :], dim=-1, keepdim=True)
            log_transmat = torch.tensor(hmm.transmat_).clamp(min=zero_eps).log()
            log_transmat = log_transmat - torch.logsumexp(log_transmat, dim=-1, keepdim=True)
            # assign same matrix to all sites
            self.eta2[k, ...] = log_transmat[None, ...]

            if self.config.debug:
                assert torch.allclose(torch.logsumexp(self.eta1[k, :], dim=-1).exp(), torch.tensor(1.))
                assert torch.allclose(torch.logsumexp(self.eta2[k, ...], dim=-1).exp(),
                                      torch.ones((self.config.chain_length - 1, self.config.n_states)))

            # hmm.means_ has shape (n_states, n_cells)
            # we aggregate over n_states, dividing by the corresponding copy number and taking a mean
            # ISSUE: we don't know the order of the states (might be any permutation of (0, ..., n_states-1)
            #   skipping mean and covar estimation
            # means[clusters == k] = hmm.means_[1:, :] / torch.arange(self.config.n_states)[1:, None]
            # precisions[clusters == k] = torch.tensor(hmm.covars_).diagonal(dim1=1, dim2=2)
            # NOTE: VariationalGaussianHMM also finds posterior estimates of params over mean and variance
            # hmmlearn.vhmm.VariationalGaussianHMM()

        # init root node
        self._init_root_skewed2()

        self.compute_filtering_probs()

    def _init_root_skewed2(self, skewness=5.):
        # skewness towards cn=2 wrt to default 1
        # e.g. skewness 5 -> cn2 will be 5 times more likely than other states in log-scale
        root_startprob = torch.ones(self.config.n_states)
        root_startprob[2] = skewness
        root_transmat = torch.ones((self.config.n_states, self.config.n_states))
        root_transmat[2, :] = skewness
        # normalize and log-transform
        self.eta1[0, :] = root_startprob - torch.logsumexp(root_startprob, dim=-1, keepdim=True)
        normalized_log_transmat = root_transmat - torch.logsumexp(root_transmat, dim=-1, keepdim=True)
        self.eta2[0, ...] = normalized_log_transmat[None, ...]  # expand for all sites 1, ..., M

    def _uniform_init(self):
        """
        Mainly used for testing.
        """
        self.eta1 = torch.ones(self.eta1.shape)
        self.eta1 = self.eta1 - torch.logsumexp(self.eta1, dim=-1, keepdim=True)
        self.eta2 = torch.ones(self.eta2.shape)
        self.eta2 = self.eta2 - torch.logsumexp(self.eta2, dim=-1, keepdim=True)

        self.compute_filtering_probs()

    def entropy(self):
        start_probs = torch.empty_like(self.eta1)
        trans_mats = torch.empty_like(self.eta2)
        if self.fixed:
            start_probs = self.single_filtering_probs[:, 0, :]
            trans_mats = self.couple_filtering_probs
        else:
            start_probs = self.eta1.exp()
            trans_mats = self.eta2.exp()

        qC_init = torch.distributions.Categorical(start_probs)
        init_entropy = qC_init.entropy().sum()
        qC = torch.distributions.Categorical(trans_mats)
        transitions_entropy = qC.entropy().sum()
        # transitions_entropy = -torch.einsum("kmij, kmij ->", self.eta2, torch.log(self.eta2))
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
            alpha_1, alpha_2 = self._exp_alpha(T_list[l], q_eps)
            cross_ent_pos_1 = torch.einsum("ki,ki->",
                                           self.single_filtering_probs[:, 0, :],
                                           alpha_1)
            cross_ent_pos_2_to_M = torch.einsum("kmij,kmij->",
                                                self.couple_filtering_probs,
                                                alpha_2)
            E_T += torch.exp(w_T_list[l] - normalizing_weight) * \
                   (cross_ent_pos_1 + cross_ent_pos_2_to_M)

        return E_T

    def neg_cross_entropy(self, T_list, w_T_list, q_eps: Union['qEpsilon', 'qEpsilonMulti']) -> float:
        # E_q[log p(C|...)]
        E_T = 0
        # v, m, j, j'
        L = len(T_list)
        for l in range(L):
            tree_CE = 0
            for a in T_list[l].edges:
                u, v = a
                arc_CE = self.neg_cross_entropy_arc(q_eps, u, v)
                tree_CE += arc_CE

            E_T += w_T_list[l] * tree_CE
        return E_T

    def neg_cross_entropy_arc(self, q_eps, u, v):
        log_h_eps0 = q_eps.h_eps0().log()
        # p(j' | j i' i)
        e_log_h_eps = q_eps.exp_log_zipping((u, v))
        cross_ent_pos_1 = torch.einsum("i,j,ji->",
                                       self.single_filtering_probs[u, 0, :],
                                       self.single_filtering_probs[v, 0, :],
                                       log_h_eps0)
        cross_ent_pos_2_to_M = torch.einsum("mik, mjl, ljki->",
                                            # u, m, i, i'
                                            self.couple_filtering_probs[u, :, :, :],
                                            # v, m, j, j'
                                            self.couple_filtering_probs[v, :, :, :],
                                            e_log_h_eps)

        return cross_ent_pos_1 + cross_ent_pos_2_to_M

    def compute_elbo(self, T_list, w_T_list, q_eps: Union['qEpsilon', 'qEpsilonMulti']) -> float:
        # unique_arcs, unique_arcs_count = tree_utils.get_unique_edges(T_list, self.config.n_nodes)
        # for (a, a_count) in zip(unique_arcs, unique_arcs_count):
        # alpha_1, alpha_2 = self.exp_alpha()
        elbo = self.neg_cross_entropy(T_list, w_T_list, q_eps) + self.marginal_entropy()
        return elbo

    def update(self, obs: torch.Tensor,
               q_eps: Union['qEpsilon', 'qEpsilonMulti'],
               q_z: 'qZ',
               q_psi: 'qPsi',
               trees,
               tree_weights) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        log q*(C) += ( E_q(mu)q(sigma)[rho_Y(Y^u, mu, sigma)] + E_q(T)[E_{C^p_u}[eta(C^p_u, epsilon)] +
        + Sum_{u,v in T} E_{C^v}[rho_C(C^v,epsilon)]] ) dot T(C^u)

        CAVI update based on the dot product of the sufficient statistic of the 
        HMM and simplified expected value over the natural parameter.
        :return:
        """
        new_eta1 = torch.zeros_like(self.eta1)
        new_eta2 = torch.zeros_like(self.eta2)

        for tree, weight in zip(trees, tree_weights):
            # compute all alpha quantities
            exp_alpha1, exp_alpha2 = self._exp_alpha(tree, q_eps)

            # init tree-specific update
            tree_eta1, tree_eta2 = self._exp_eta(obs, tree, q_eps, q_z, q_psi)
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
        # logging.debug("- copy number updated")
        super().update()

    def update_params(self, eta1, eta2):
        lrho = torch.tensor(self.config.step_size).log()
        l1mrho = torch.tensor(1. - self.config.step_size).log()
        # numerically stable sum over prob vectors
        new_eta1 = torch.logaddexp(self.eta1 + l1mrho, eta1 + lrho)
        new_eta2 = torch.logaddexp(self.eta2 + l1mrho, eta2 + lrho)
        # add normalization step
        self.eta1 = new_eta1 - new_eta1.logsumexp(dim=1, keepdim=True)
        self.eta2 = new_eta2 - new_eta2.logsumexp(dim=3, keepdim=True)
        if self.config.debug:
            assert np.allclose(self.eta1.logsumexp(dim=1).exp(), 1.)
            assert np.allclose(self.eta2.logsumexp(dim=3).exp(), 1.)
        return self.eta1, self.eta2

    def _exp_eta(self, obs: torch.Tensor, tree: nx.DiGraph,
                 q_eps: Union['qEpsilon', 'qEpsilonMulti'],
                 q_z: 'qZ',
                 q_psi: 'qPsi') -> Tuple[torch.Tensor, torch.Tensor]:
        """Expectation of natural parameter vector \\eta

        Parameters
        ----------
        tree : nx.DiGraph
            Tree on which expectation is taken (e.g. sampled tree)
        q_eps : qEpsilon
            Variational distribution object of epsilon parameter
        q_z : VariationalDistribution
            Variational distribution object of cell assignment 
        q_psi : qMuTau
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
        e_eta1_m = torch.einsum('nv,nmvi->vmi',
                                q_z.exp_assignment(),
                                q_psi.exp_log_emission(obs))
        # eta_1_iota(1, i)
        e_eta1[inner_nodes, :] = torch.einsum('pj,ij->pi',
                                              self.single_filtering_probs[
                                              [next(tree.predecessors(u)) for u in inner_nodes], 0, :],
                                              q_eps.h_eps0().log()) + \
                                 e_eta1_m[inner_nodes, 0, :]
        # eta_2_kappa(m, i, i')
        if not isinstance(q_eps, qEpsilonMulti):
            e_eta2[...] = torch.einsum('pmjk,hikj->pmih',
                                       self.couple_filtering_probs,
                                       q_eps.exp_log_zipping()) + \
                          e_eta1_m[:, 1:, None, :]
        else:
            edges_mask = [[p for p, _ in tree.edges], [v for _, v in tree.edges]]
            e_eta2[edges_mask[1], ...] = torch.einsum('pmjk,phikj->pmih',
                                                      self.couple_filtering_probs[edges_mask[0], ...],
                                                      torch.stack([q_eps.exp_log_zipping(e) for e in tree.edges])) + \
                                         e_eta1_m[edges_mask[1], 1:, None, :]

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
        # alpha is not defined for root node
        e_alpha1[0, ...] = -torch.inf
        e_alpha2[0, ...] = -torch.inf

        # alpha_iota(m, i)
        # as in the write-up, then it's split and e_alpha12[:, 1:, :] is
        # incorporated into e_alpha2
        e_alpha12 = torch.einsum('wmj,ji->wmi', self.single_filtering_probs, q_eps.h_eps0().log())

        e_alpha1[...] = e_alpha12[:, 0, :]  # first site goes to initial state

        # alpha_kappa(m, i, i')
        # similar to eta2 but with inverted indices in zipping
        if not isinstance(q_eps, qEpsilonMulti):
            e_alpha2 = torch.einsum('wmjk,kjhi->wmih',
                                    self.couple_filtering_probs,
                                    q_eps.exp_log_zipping((0, 0))) + \
                       e_alpha12[:, 1:, None, :]
        else:
            edges_mask = [[p for p, _ in tree.edges], [v for _, v in tree.edges]]
            e_alpha2[edges_mask[1], ...] = torch.einsum('wmjk,wkjhi->wmih',
                                                        self.couple_filtering_probs[edges_mask[1], ...],
                                                        torch.stack([q_eps.exp_log_zipping(e) for e in tree.edges])) + \
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
        small_eps = 1e-8
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
            # avoid error propagation with normalization step
            log_couple[:, m, ...] = log_couple[:, m, ...] - log_couple[:, m, ...].logsumexp(dim=(1, 2), keepdim=True)
            # then marginalize over X_m to obtain P(X_m+1)
            log_single[:, m + 1, :] = torch.logsumexp(log_couple[:, m, ...], dim=1)

            if self.config.debug:
                assert np.allclose(log_single[:, m + 1, :].exp().sum(dim=1), 1.)
                assert np.allclose(log_couple[:, m, ...].exp().sum(dim=(1, 2)), 1.)

        self.single_filtering_probs = torch.exp(log_single).clamp(min=small_eps, max=1. - small_eps)
        self.couple_filtering_probs = torch.exp(log_couple).clamp(min=small_eps, max=1. - small_eps)

        if self.config.debug:
            assert np.allclose(self.single_filtering_probs.sum(dim=2), 1.)
            assert np.allclose(self.couple_filtering_probs.sum(dim=(2, 3)), 1.)

        return self.single_filtering_probs, self.couple_filtering_probs

    def get_viterbi(self) -> torch.Tensor:
        """
        Computes the Viterbi path of each node's non-homogeneous Markov chain.
        Follows the pseudo-code in
        https://en.wikipedia.org/wiki/Viterbi_algorithm
        but emissions are encoded in the transitions probabilities.

        Returns tensor of shape (n_nodes, chain_length) dtype=long
        """

        M = self.config.chain_length

        init_probs_qu = torch.exp(self.eta1 - torch.logsumexp(self.eta1, dim=1, keepdim=True))
        transition_probs = torch.exp(self.eta2 -
                                     torch.logsumexp(self.eta2, dim=3, keepdim=True))
        t1 = torch.empty((self.config.n_nodes, self.config.n_states, M))
        t2 = torch.empty((self.config.n_nodes, self.config.n_states, M))
        # init first site
        t1[:, :, 0] = init_probs_qu
        t2[:, :, 0] = 0.
        # forward
        for m in range(1, M):
            t1[:, :, m], t2[:, :, m] = torch.max(t1[:, :, m - 1, None] * transition_probs[:, m - 1, ...], dim=1)

        # init backtrack
        zm = torch.empty((self.config.n_nodes, M), dtype=torch.long)
        zm[:, M - 1] = t1[:, :, M - 1].max(dim=1)[1]
        # backward
        for m in reversed(range(1, M)):
            nodes_range = torch.arange(self.config.n_nodes)
            zm[:, m - 1] = t2[nodes_range, zm[nodes_range, m], m]

        return zm

    def __str__(self):
        torch.set_printoptions(precision=3)
        # summary for qc
        summary = ["[qC summary]"]
        if self.fixed:
            summary[0] += " - True Dist"
            k = max(30, self.config.chain_length)  # first k sites
            summary.append(f"-cn profile\n{self.true_params['c'][:, :k]}")
        else:
            max_entropy = torch.special.entr(torch.ones(self.config.n_states) / self.config.n_states).sum()
            # print first k cells assignments and their uncertainties - entropy normalized by max value
            summary.append(f"-cn profile")
            k = min(10, self.config.n_nodes)  # first k clones
            m = min(30, self.config.chain_length)  # first m sites
            for c in range(k):
                cn_prof = self.single_filtering_probs[c, :m, :].argmax(dim=1)
                summary.append(f"\t{c}:\t{cn_prof}")
                uncertainty = torch.special.entr(self.single_filtering_probs[c, :m, :]).sum(dim=1) / max_entropy
                summary.append(f"\t\t{uncertainty}")

        return os.linesep.join(summary)

    def get_checkpoint(self):
        return {"eta1": self.eta1, "eta2": self.eta2}
    # TODO: continue

    def smooth_etas(self):
        lag = 2
        for k in range(self.config.n_nodes):
            for m in range(2, self.config.chain_length - 3):
                current_state = self.eta2[k, m, :, :].argmax(dim=-1)
                prev_state = self.eta2[k, m-1, :, :].argmax(dim=-1)
                next_state = self.eta2[k, m+1, :, :].argmax(dim=-1)
                if (prev_state != current_state).any() and \
                        (self.eta2[k, m-2, :, :].argmax(dim=-1) == prev_state).any() and \
                        (current_state != next_state).any() and \
                        (self.eta2[k, m+2, :, :].argmax(dim=-1) == next_state).any():  # lag 2 outlier
                    
                    if torch.abs(current_state - prev_state).sum() < torch.abs(current_state - next_state).sum():
                        self.eta2[k, m, :, :] = self.eta2[k, m-1, :, :]
                    else:
                        self.eta2[k, m, :, :] = self.eta2[k, m+1, :, :]

class qCMultiChrom(VariationalDistribution):

    def __init__(self, config: Config, true_params=None):
        """
        Wrapper class for multiple Markov chains, each one for each chromosome.
        Parameters
        ----------
        config: Config, configuration object with chromosome_indices attr properly set
        true_params: dict, contains "c" key which is a torch.Tensor of shape (n_nodes, chain_length) with
            copy number integer values
        """
        super().__init__(config, fixed=true_params is not None)

        self.qC_list: List[qC] = []
        self.n_chr = config.n_chromosomes
        self.chr_start_points = [0] + config.chromosome_indexes + [config.chain_length]
        self.M_cr = []
        for i in range(self.n_chr):
            M_chr_i = self.chr_start_points[i + 1] - self.chr_start_points[i]
            config_i = copy.deepcopy(self.config)
            config_i.chain_length = M_chr_i
            self.qC_list.append(qC(config_i, true_params=true_params))

        self._single_filtering_probs = torch.empty((config.n_nodes, config.chain_length, config.n_states))
        self._couple_filtering_probs = torch.empty(
            (config.n_nodes, config.chain_length - self.n_chr, config.n_states, config.n_states))

        # eta1 = log(pi) - log initial states probs
        self._eta1: torch.Tensor = torch.empty((config.n_nodes, config.n_states))
        # eta2 = log(phi) - log transition probs
        self._eta2: torch.Tensor = torch.empty_like(self._couple_filtering_probs)

        # validate true params
        if true_params is not None:
            assert "c" in true_params
            raise NotImplementedError("qCMultiChrom has not been tested for fixed distr, might not work as expected")
        self.true_params = true_params

        # define dist param names
        self.params_history["single_filtering_probs"] = []
        # # not needed at the moment
        # self.params_history["couple_filtering_probs"] = []

    def update(self, obs: torch.Tensor,
               q_eps: Union['qEpsilon', 'qEpsilonMulti'],
               q_z: 'qZ',
               q_psi: 'qPsi',
               trees,
               tree_weights) -> Tuple[torch.Tensor, torch.Tensor]:

        for i, qc in enumerate(self.qC_list):
            chr_i_start = self.chr_start_points[i]
            chr_i_end = self.chr_start_points[i + 1]
            qc.update(obs[chr_i_start:chr_i_end, :], q_eps, q_z, q_psi, trees, tree_weights)

        self.compute_filtering_probs()

    def compute_filtering_probs(self):
        for i, qc in enumerate(self.qC_list):
            single_i, couple_i = qc.compute_filtering_probs()
            m_start = self.chr_start_points[i]
            m_end = self.chr_start_points[i + 1]
            self._single_filtering_probs[:, m_start:m_end, :] = single_i
            self._couple_filtering_probs[:, m_start - i:m_end - i - 1, :, :] = couple_i

    @property
    def single_filtering_probs(self):
        if self.fixed:
            small_eps = 1e-5
            cn_profile = self.true_params["c"]
            true_sfp = torch_functional.one_hot(cn_profile,
                                                num_classes=self.config.n_states).float().clamp(min=small_eps)
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
            small_eps = 1e-5
            cn_profile = self.true_params["c"]
            # map node specific copy number profile to pair marginal probabilities
            # almost all prob mass is given to the true copy number combinations
            true_cfp = torch.zeros_like(self._couple_filtering_probs)
            for u in range(self.config.n_nodes):
                true_cfp[u, torch.arange(self.config.chain_length - 1),
                         cn_profile[u, :-1], cn_profile[u, 1:]] = 1.
            # add epsilon and normalize
            self._couple_filtering_probs[...] = true_cfp.clamp(min=small_eps)
            self._couple_filtering_probs /= self._couple_filtering_probs.sum(dim=(2, 3), keepdim=True)
            if self.config.debug:
                assert torch.allclose(self._couple_filtering_probs.sum(dim=(2, 3)),
                                      torch.ones((self.config.n_nodes, self.config.chain_length - 1)))

        return self._couple_filtering_probs

    @couple_filtering_probs.setter
    def couple_filtering_probs(self, cfp):
        if self.fixed:
            logging.warning('Trying to re-set qc attribute when it should be fixed')

        self._couple_filtering_probs[...] = cfp

    def initialize(self, method='random', **kwargs):
        for qc in self.qC_list:
            if method == 'baum-welch':
                qc._baum_welch_init(**kwargs)
            elif method == 'bw-cluster':
                qc._init_bw_cluster_data(**kwargs)
            elif method == 'random':
                qc._random_init()
            elif method == 'uniform':
                qc._uniform_init()
            else:
                raise ValueError(f'method `{method}` for qC initialization is not implemented')
            self.compute_filtering_probs()

        return super().initialize(**kwargs)

    def compute_elbo(self, T_list, w_T_list, q_eps: Union['qEpsilon', 'qEpsilonMulti']) -> float:
        elbo = 0
        for qc in self.qC_list:
            elbo += qc.compute_elbo(T_list, w_T_list, q_eps)
        return elbo

    # TODO: implement __str__


# cell assignments
class qZ(VariationalDistribution):
    def __init__(self, config: Config, true_params=None):
        super().__init__(config, true_params is not None)

        self._pi = torch.empty((config.n_cells, config.n_nodes))

        self.kmeans_labels = torch.empty(config.n_cells, dtype=torch.long)
        if true_params is not None:
            assert "z" in true_params
        self.true_params = true_params

        self.params_history["pi"] = []

    @property
    def pi(self):
        if self.fixed:
            true_z = self.true_params['z']
            # set prob of a true assignment to 1
            self._pi[torch.arange(self.config.n_cells), true_z] = 1.
        return self._pi

    @pi.setter
    def pi(self, pi):
        if self.fixed:
            logging.warning('Trying to re-set qc attribute when it should be fixed')
        self._pi[...] = pi

    def initialize(self, z_init: str = 'random', **kwargs):
        if z_init == 'random':
            self._random_init()
        elif z_init == 'uniform':
            self._uniform_init()
        elif z_init == 'fixed':
            self._init_with_values(**kwargs)
        elif z_init == 'kmeans':
            self._kmeans_init(**kwargs)
        else:
            raise ValueError(f'method `{z_init}` for qZ initialization is not implemented')
        return super().initialize(**kwargs)

    def _random_init(self):
        # sample from a Dirichlet
        self.pi[...] = torch.distributions.Dirichlet(torch.ones_like(self.pi)).sample()

    def _init_with_values(self, pi_init):
        self.pi[...] = pi_init

    def _uniform_init(self):
        # initialize to uniform probs among nodes
        self.pi[...] = torch.ones_like(self.pi) / self.config.n_nodes

    def _kmeans_init(self, obs, **kwargs):
        # TODO: find a soft k-means version
        # https://github.com/omadson/fuzzy-c-means
        logging.debug("Running k-means for z init")
        eps = 1e-4
        N = self.config.n_cells
        M = self.config.chain_length
        obs = obs.T if obs.shape == (N, M) else obs
        m_obs = obs.mean(dim=0, keepdim=True)
        sd_obs = obs.std(dim=0, keepdim=True)
        # standardize to keep pattern
        scaled_obs = (obs - m_obs) / sd_obs.clamp(min=eps)
        kmeans = KMeans(n_clusters=self.config.n_nodes, random_state=0).fit(scaled_obs.T)
        m_labels = kmeans.labels_
        self.kmeans_labels[...] = torch.tensor(m_labels).long()
        self.pi[...] = torch.nn.functional.one_hot(self.kmeans_labels, num_classes=self.config.n_nodes)

    def _kmeans_per_site_init(self, obs, qmt: 'qMuTau'):
        M, N = obs.shape
        K = self.config.n_nodes
        A = self.config.n_states
        for m in range(M):
            kmeans = KMeans(n_clusters=K, random_state=0).fit(obs[m, :])
            m_labels = kmeans.labels_
        raise NotImplemented("kmeans_per_site_init not complete")

    def update(self, qpsi: 'qPsi', qc: 'qC', qpi: 'qPi', obs: torch.Tensor):
        """
        q(Z) is a Categorical with probabilities pi^{*}, where pi_k^{*} = exp(gamma_k) / sum_K exp(gamma_k)
        gamma_k = E[log \pi_k] + sum_{m,j} q(C_m^k = j) E_{\mu, \tau}[D_{nmj}]
        :param Y: observations
        :param q_C_marginal:
        :param q_pi_dist:
        :param qpsi:
        :return:
        """
        # single_filtering_probs: q(Cmk = j), shape: K x M x J
        qc_kmj = qc.single_filtering_probs
        # expected log pi
        e_logpi = qpi.exp_log_pi()
        # Dnmj
        d_nmj = qpsi.exp_log_emission(obs)

        # op shapes: k + S_mS_j mkj nmj -> nk
        gamma = e_logpi + torch.einsum('kmj,nmkj->nk', qc_kmj, d_nmj)
        T = self.config.annealing
        gamma = gamma * 1 / T
        pi = torch.softmax(gamma, dim=1)
        new_pi = self.update_params(pi)
        # logging.debug("- z updated")
        super().update()

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

    def neg_cross_entropy(self, qpi: 'qPi') -> float:
        e_logpi = qpi.exp_log_pi()
        return torch.einsum("nk, k -> ", self.pi, e_logpi)

    def entropy(self) -> float:
        return torch.special.entr(self.pi).sum()

    def compute_elbo(self, qpi: 'qPi') -> float:
        return self.neg_cross_entropy(qpi) + self.entropy()

    def __str__(self):
        np.set_printoptions(precision=3, suppress=True)
        # summary for qz
        summary = ["[qZ summary]"]
        if self.fixed:
            summary[0] += " - True Dist"
            k = max(20, self.config.n_cells)  # first k cells
            summary.append("-cell assignment\t" + self.true_params['z'][:k])
        else:
            max_entropy = torch.special.entr(torch.ones(self.config.n_nodes) / self.config.n_nodes).sum()
            # print first k cells assignments and their uncertainties - entropy normalized by max value
            summary.append(f"-cell assignment (uncertainty, i.e. 1=flat, 0=peaked)\n")
            k = min(10, self.config.n_cells)  # first k cells
            for c in range(k):
                cell_clone = self.pi[c, :].argmax()
                norm_entropy = torch.special.entr(self.pi[c, :]).sum() / max_entropy
                summary.append(f"\t\tcell {c}: clone {cell_clone} ({norm_entropy.item():.3f})")

        return os.linesep.join(summary)


# topology
class qT(VariationalDistribution):

    def __init__(self, config: Config, true_params=None):
        super().__init__(config, fixed=true_params is not None)
        # weights are in log-form
        # so that tree.size() is log_prob of tree (sum of log_weights)
        self.log_g_t = torch.empty((config.wis_sample_size,))
        self.log_w_t = torch.zeros((config.wis_sample_size,))
        self.nx_trees_sample = []
        self._weighted_graph = nx.DiGraph()
        self._weighted_graph.add_edges_from([(u, v)
                                             for u, v in itertools.permutations(range(config.n_nodes), 2)
                                             if u != v and v != 0])

        if true_params is not None:
            assert 'tree' in true_params
        self.true_params = true_params

        self.params_history["weight_matrix"] = []
        self.params_history["trees_sample_newick"] = []
        self.params_history["trees_sample_weights"] = []

    @property
    def weighted_graph(self):
        return self._weighted_graph

    @property
    def weight_matrix(self) -> np.ndarray:
        return nx.to_numpy_array(self.weighted_graph)

    @property
    def trees_sample_newick(self) -> np.ndarray:
        return np.array([tree_utils.tree_to_newick(t) for t in self.nx_trees_sample], dtype='S')

    @property
    def trees_sample_weights(self) -> np.ndarray:
        return self.log_w_t.exp().data.cpu().numpy()

    def _init_from_matrix(self, matrix, **kwargs):
        for e in self._weighted_graph.edges:
            self._weighted_graph.edges[e]['weight'] = torch.tensor(matrix[e])
        # run sampling to store first sampled tree list and weights
        self.get_trees_sample()

    def initialize(self, method='random', **kwargs):
        if method == 'random':
            # rooted graph with random weights in (0, 1) - log transformed
            self.init_fc_graph()
        elif method == 'matrix':
            self._init_from_matrix(**kwargs)
        else:
            raise ValueError(f'method `{method}` for qT initialization is not implemented')

        return super().initialize(**kwargs)

    def neg_cross_entropy(self):
        # sampled trees are not needed here
        # negative cross_entropy = sum_t q(t) log p(t) = log p(t) = - log | T |
        # (number of possible labeled rooted trees)
        # it's equal to Cayley's formula since we fix root = 0
        return -math_utils.cayleys_formula(self.config.n_nodes, log=True)

    def entropy(self, trees, weights):
        # H(qt) = - E_qt[ log qt(T) ] \approx -1/n sum_i w(T_i) log qt(T_i)
        # TODO: how do we compute the entropy if we don't know the normalized q(T)?
        #   t.size() computes \log\tilde q(T),
        entropy = 0.
        for i, t in enumerate(trees):
            log_qt = t.size(weight='weight')
            entropy -= weights[i] * log_qt
        return entropy / sum(weights)

    def compute_elbo(self, trees: list | None = None, weights: torch.Tensor | None = None) -> float:
        """
Computes partial elbo for qT from the same trees-sample used for
other elbos such as qC.
        Args:
            trees: list of nx.DiGraph
            weights: list of weights as those in the qT.get_trees_sample() output
        Returns:
            float, value of ELBO for qT
        """
        if trees is None or weights is None:
            trees = self.nx_trees_sample
            weights = self.log_w_t.exp()
        elbo = self.neg_cross_entropy() + self.entropy(trees, weights)
        # convert to python float
        return elbo.item()

    def update(self, qc: qC, qeps: Union['qEpsilon', 'qEpsilonMulti']):
        # q_T = self.update_CAVI(T_list, qc, qeps)
        self.update_graph_weights(qc, qeps)
        # logging.debug("- tree updated")
        super().update()

    def update_params(self, new_weights: torch.Tensor):
        rho = self.config.step_size  # TODO: step size not applicable here?
        prev_weights = torch.tensor([w for u, v, w in self._weighted_graph.edges.data('weight')])
        stepped_weights = (1 - rho) * prev_weights + rho * new_weights

        for i, (u, v, weight) in enumerate(self._weighted_graph.edges.data('weight')):
            self._weighted_graph.edges[u, v]['weight'] = stepped_weights[i]
        return self._weighted_graph.edges.data('weight')

    def update_graph_weights(self, qc: qC, qeps: Union['qEpsilon', 'qEpsilonMulti']):
        all_edges = [(u, v) for u, v in self._weighted_graph.edges]
        new_log_weights = {}
        for u, v in all_edges:
            new_log_weights[u, v] = torch.einsum('mij,mkl,jilk->', qc.couple_filtering_probs[u],
                                                 qc.couple_filtering_probs[v], qeps.exp_log_zipping((u, v)))
        # chain length determines how large log-weights are
        # while they should be length invariant
        # FIXME: avoid this hack
        # TODO: implement tempering (check tempered/annealing in VI)
        w_tensor = torch.tensor(list(new_log_weights.values())) / self.config.chain_length
        self.update_params(w_tensor)

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
            E_CuCveps[u, v] = torch.einsum('mij, mkl, lkji -> ', q_C_pairwise_marginals[u], q_C_pairwise_marginals[v],
                                           E_eps_h)

        for k, T in enumerate(T_list):
            for u, v in T.edges:
                log_q_T_tensor[k] += E_CuCveps[u, v]

        return log_q_T_tensor

    def init_fc_graph(self):
        # random initialization of the fully connected graph over the clones
        for e in self._weighted_graph.edges:
            self._weighted_graph.edges[e]['weight'] = torch.rand(1)[0].log()
        # run sampling to store first sampled tree list and weights
        self.get_trees_sample()

    def get_trees_sample(self, alg: str = 'laris', sample_size: int = None,
                         torch_tensor: bool = False, log_scale: bool = False,
                         add_log_g=False) -> (list, list | torch.Tensor):
        """
Sample trees from q(T) with importance sampling.
        Args:
            alg: string, chosen in ['random' | 'laris']
            sample_size: number of trees to be sampled. If None, sample_size is taken from the configuration
                object
        Returns:
            list of nx.DiGraph arborescences and list of related weights for computing expectations
            The weights are the result of the operation q'(T) / g'(T) where
                - q'(T) is the unnormalized probability under q(T), product of arc weights
                - g'(T) is the probability of the sample, product of Bernoulli trials (also unnormalized)
            the output weights are also normalized among the sample

        Parameters
        ----------
        torch_tensor: bool, if true, weights are returned as torch.Tensor
        """
        # e.g.:
        # trees = edmonds_tree_gen(self.config.is_sample_size)
        # trees = csmc_tree_gen(self.config.is_sample_size)
        trees = []
        l = self.config.wis_sample_size if sample_size is None else sample_size
        log_weights = torch.empty(l)
        log_gs = torch.empty(l)
        if self.fixed:
            trees = [self.true_params['tree']] * l
            log_weights[...] = torch.ones(l)

        elif alg == "random":
            trees = [nx.random_tree(self.config.n_nodes, create_using=nx.DiGraph)
                     for _ in range(l)]
            log_weights[...] = torch.ones(l)
            for t in trees:
                nx.set_edge_attributes(t, np.random.rand(len(t.edges)), 'weight')

        elif alg == "laris":
            for i in range(l):
                t, log_g = sample_arborescence_from_weighted_graph(self.weighted_graph)
                trees.append(t)
                log_q = t.size(weight='weight')  # unnormalized q(T)
                log_weights[i] = log_q - log_g
                log_gs[i] = log_g
                # get_trees_sample can be called with arbitrary sample_size
                # e.g. in case of evaluation we might want more than config.wis_sample_size trees
                # this avoids IndexOutOfRange error
                if i < self.config.wis_sample_size:
                    self.log_g_t[i] = log_g.detach().clone()
                    self.log_w_t[i] = log_weights[i]
            # the weights are normalized
            # TODO: aggregate equal trees and adjust their weights accordingly
            log_weights[...] = log_weights - torch.logsumexp(log_weights, dim=-1)
        else:
            raise ValueError(f"alg '{alg}' is not implemented, check the documentation")

        # only first trees are saved (see comment above)
        # FIXME: this is just a temporary fix for sample_size param being different than config.wis_sample_size
        min_size = min(self.config.wis_sample_size, l)
        self.nx_trees_sample = trees[:min_size]
        out_weights = log_weights
        if not log_scale:
            out_weights = torch.exp(log_weights)
            out_log_g = torch.exp(log_gs)
        if not torch_tensor:
            out_weights = out_weights.tolist()
            out_log_g = log_gs.tolist()

        if add_log_g:
            return trees, out_weights, out_log_g
        else:
            return trees, out_weights

    def enumerate_trees(self) -> (list, torch.Tensor):
        """
Enumerate all labeled trees (rooted in 0) with their probability
q(T) associated to the weighted graph which represents the current state
of the variational distribution over the topology.
        Returns
        -------
        tuple with list of nx.DiGraph (trees) and tensor with normalized log-probabilities
        """
        c = 0
        tot_trees = math_utils.cayleys_formula(self.config.n_nodes)
        trees = []
        trees_log_prob = torch.empty(tot_trees)
        for pruf_seq in itertools.product(range(self.config.n_nodes), repeat=self.config.n_nodes - 2):
            unrooted_tree = nx.from_prufer_sequence(list(pruf_seq))
            rooted_tree = nx.dfs_tree(unrooted_tree, 0)
            rooted_tree_with_weigths = copy.deepcopy(self.weighted_graph)
            rooted_tree_with_weigths.remove_edges_from(e for e in self.weighted_graph.edges
                                                       if e not in rooted_tree.edges)
            trees.append(rooted_tree_with_weigths)
            trees_log_prob[c] = rooted_tree_with_weigths.size(weight='weight')
            c += 1

        trees_log_prob[...] = trees_log_prob - trees_log_prob.logsumexp(dim=0)
        assert tot_trees == c
        return trees, trees_log_prob

    def __str__(self):
        np.set_printoptions(precision=3, suppress=True)
        # summary for qt
        summary = ["[qT summary]"]
        if self.fixed:
            summary[0] += " - True Dist"
            summary.append("-tree\t" + tree_utils.tree_to_newick(self.true_params['tree']))
        else:
            summary.append(f"-adj matrix\t{nx.to_numpy_array(self._weighted_graph)}")
            summary.append(f"-sampled trees:")
            qdist = self.get_pmf_estimate()
            # TODO: sort in get_pmf_estimate
            display_num = min(10, len(qdist.keys()))
            for t_nwk in sorted(qdist, key=qdist.get, reverse=True)[:display_num]:
                summary.append(f"\t\t{t_nwk} | {qdist[t_nwk]:.4f}")
            summary.append(f"partial ELBO\t{self.compute_elbo():.2f}")

        return os.linesep.join(summary)

    def get_pmf_estimate(self, normalized: bool = False, n: int = 0, desc_sorted: bool = False) -> dict:
        """
        Returns
        -------
        dict with values (newick_tree: sum of importance_weights)
        """
        qdist = {}
        trees, log_w_t = self.nx_trees_sample, self.log_w_t
        if n > 0:
            trees, log_w_t_list = self.get_trees_sample(sample_size=n, log_scale=True)
            log_w_t = torch.tensor(log_w_t_list)
        for t, log_w in zip(trees, log_w_t):
            # build a pmf from the sample by summing up the importance weights
            w = log_w.exp()
            newick = tree_utils.tree_to_newick(t)
            if newick not in qdist:
                qdist[newick] = 0.
            qdist[newick] += w

        if normalized:
            norm_const = sum(qdist.values())
            for t in qdist.keys():
                qdist[t] /= norm_const

        if desc_sorted:
            qdist = dict(sorted(qdist.items(), key=lambda it: it[1], reverse=True))

        return qdist


# edge distance (single eps for all nodes)
class qEpsilon(VariationalDistribution):

    def __init__(self, config: Config, alpha_0: float = 1., beta_0: float = 1.):
        super().__init__(config)

        self.alpha_prior = torch.tensor(alpha_0, dtype=torch.float32)
        self.beta_prior = torch.tensor(beta_0, dtype=torch.float32)
        self.alpha = torch.tensor(alpha_0, dtype=torch.float32)
        self.beta = torch.tensor(beta_0, dtype=torch.float32)
        self._exp_log_zipping = None

        self.params_history["alpha"] = []
        self.params_history["beta"] = []

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
        comut_mask = torch.zeros((A, A, A, A))
        anti_sym_mask = torch.zeros((A, A, A, A))
        # TODO: Find effecient way of indexing i-j = k-l
        for i, j, k, l in itertools.combinations_with_replacement(range(A), 4):
            if i - j == k - l:
                comut_mask[i, j, k, l] = 1
            else:
                anti_sym_mask[i, j, k, l] = 1
        return comut_mask, anti_sym_mask

    def compute_elbo(self) -> float:
        return super().compute_elbo()

    def update(self, T_list, w_T, q_C_pairwise_marginals):
        self.update_CAVI(T_list, w_T, q_C_pairwise_marginals)
        # logging.debug("- eps updated")
        super().update()

    def update_CAVI(self, T_list: list, w_T: torch.Tensor, q_C_pairwise_marginals: torch.Tensor):
        N, M, A, A = q_C_pairwise_marginals.size()
        alpha = self.alpha_prior
        beta = self.beta_prior
        unique_edges, unique_edges_count = tree_utils.get_unique_edges(T_list, N_nodes=N)

        E_CuCv_a = torch.zeros((N, N))
        E_CuCv_b = torch.zeros((N, N))
        comut_mask, no_comut_mask, abs_state_mask = get_zipping_mask(A)
        for uv in unique_edges:
            u, v = uv
            E_CuCv_a[u, v] = torch.einsum('mij, mkl, lkji -> ', q_C_pairwise_marginals[u], q_C_pairwise_marginals[v],
                                          (no_comut_mask).float())
            E_CuCv_b[u, v] = torch.einsum('mij, mkl, lkji -> ', q_C_pairwise_marginals[u], q_C_pairwise_marginals[v],
                                          comut_mask.float())

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
            heps0_arr = self.config.eps0 / (self.config.n_states - 1) * torch.ones(
                (self.config.n_states, self.config.n_states))
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
            self._exp_log_zipping = torch.empty_like((self.config.n_states,) * 4)
            # bool tensor with True on [j', j, i', i] where j'-j = i'-i (comutation)
            comut_mask, no_comut_mask, abs_state_mask = get_zipping_mask(self.config.n_states)

            # exp( E_CuCv[ log( 1 - eps) ] )
            # switching to exponential leads to easier normalization step
            # (same as the one in `h_eps()`)
            exp_E_log_1meps = comut_mask * torch.exp(torch.digamma(self.beta) -
                                                     torch.digamma(self.alpha + self.beta))
            exp_E_log_eps = (1. - exp_E_log_1meps.sum(dim=0)) / torch.sum(no_comut_mask, dim=0)
            self._exp_log_zipping[...] = exp_E_log_eps * (no_comut_mask) + exp_E_log_1meps
            if self.config.debug:
                assert torch.allclose(torch.sum(self._exp_log_zipping, dim=0), torch.ones_like(self._exp_log_zipping))
            self._exp_log_zipping[...] = self._exp_log_zipping.log()

        return self._exp_log_zipping


# edge distance (multiple eps, one for each arc)
class qEpsilonMulti(VariationalDistribution):

    def __init__(self, config: Config, alpha_prior: float = 1., beta_prior: float = 5., gedges=None,
                 true_params=None):
        super().__init__(config, true_params is not None)

        # so that only admitted arcs are present (and self arcs such as v->v are not accessible)
        self.alpha_prior = torch.tensor(alpha_prior)
        self.beta_prior = torch.tensor(beta_prior)
        if gedges is None:
            # one param for every arc except self referenced and v -> root for any v
            gedges = [(u, v) for u, v in itertools.product(range(config.n_nodes),
                                                           range(config.n_nodes)) if v != 0 and u != v]
        self._alpha_dict = {e: torch.empty(1) for e in gedges}
        self._beta_dict = {e: torch.empty(1) for e in gedges}

        co_mut_mask, no_co_mut_mask, abs_state_mask = self.create_masks(config.n_states)
        self.co_mut_mask = co_mut_mask
        self.no_co_mut_mask = no_co_mut_mask
        self.abs_state_mask = abs_state_mask

        if true_params is not None:
            assert "eps" in true_params
        self.true_params = true_params

        self.params_history["alpha"] = []
        self.params_history["beta"] = []

    @property
    def alpha_dict(self):
        return self._alpha_dict

    @alpha_dict.setter
    def alpha_dict(self, a: dict):
        for e, w in a.items():
            self._alpha_dict[e] = w

    @property
    def beta_dict(self):
        return self._beta_dict

    @beta_dict.setter
    def beta_dict(self, b: dict):
        for e, w in b.items():
            self._beta_dict[e] = w

    @property
    def alpha(self):
        """
        Numpy array version of the alpha parameter. To be used as a checkpoint.
        """
        return edge_dict_to_matrix(self._alpha_dict, self.config.n_nodes)

    @property
    def beta(self):
        """
        Numpy array version of the beta parameter. To be used as a checkpoint.
        """
        return edge_dict_to_matrix(self._beta_dict, self.config.n_nodes)

    def update_params(self, alpha: dict, beta: dict):
        rho = self.config.step_size
        for e in self.alpha_dict.keys():
            self._alpha_dict[e] = (1 - rho) * self.alpha_dict[e] + rho * alpha[e]
            self._beta_dict[e] = (1 - rho) * self.beta_dict[e] + rho * beta[e]
        return self.alpha_dict, self.beta_dict

    def update(self, tree_list: list, tree_weights: torch.Tensor, qc: qC):
        self.update_CAVI(tree_list, tree_weights, qc)
        super().update()

    def update_CAVI(self, tree_list: list, tree_weights: torch.Tensor, qc: qC):
        cfp = qc.couple_filtering_probs
        K, M, A, A = cfp.shape
        new_alpha = {(u, v): self.alpha_prior.detach().clone() for u, v in self._alpha_dict.keys()}
        new_beta = {(u, v): self.beta_prior.detach().clone() for u, v in self._beta_dict.keys()}
        unique_edges, unique_edges_count = tree_utils.get_unique_edges(tree_list, N_nodes=K)

        # check how many edges are effectively updated
        # after some iterations (might be very few)
        if len(unique_edges) < len(new_alpha):
            logging.debug(f"\t[qEps] updating {len(unique_edges)}/{len(new_alpha)} edges,"
                          f" consider increasing trees sample size")

        # E_T[ sum_m sum_{not A} Cu Cv ]
        exp_cuv_a = {}
        # E_T[ sum_m sum_{A} Cu Cv ]
        exp_cuv_b = {}
        comut_mask = self.co_mut_mask  #get_zipping_mask(A)
        no_comut_mask = self.no_co_mut_mask  #get_zipping_mask(A)
        abs_state_mask = self.abs_state_mask  #get_zipping_mask(A)
        for u, v in unique_edges:
            exp_cuv_a[u, v] = torch.einsum('mij, mkl, lkji -> ',
                                           cfp[u],
                                           cfp[v],
                                           (no_comut_mask).float())
            exp_cuv_b[u, v] = torch.einsum('mij, mkl, lkji -> ',
                                           cfp[u],
                                           cfp[v],
                                           comut_mask.float())

        for k, t in enumerate(tree_list):
            for e in t.edges:
                ww = tree_weights[k]
                # only change the values related to the tree edges
                new_alpha[e] += ww * exp_cuv_a[e]
                new_beta[e] += ww * exp_cuv_b[e]

        self.update_params(new_alpha, new_beta)
        return new_alpha, new_beta

    def _set_equal_params(self, eps_alpha: float, eps_beta: float):
        for e in self.alpha_dict.keys():
            self.alpha_dict[e] = torch.tensor(eps_alpha)
            self.beta_dict[e] = torch.tensor(eps_beta)

    def initialize(self, method='random', **kwargs):
        if method == 'fixed':
            self._fixed_init(**kwargs)
        elif method == 'fixed-equal':
            self._set_equal_params(**kwargs)
        elif method == 'uniform':
            self._uniform_init()
        elif method == 'random':
            self._random_init(**kwargs)
        elif method == 'non_mutation':
            self._non_mutation_init()
        else:
            raise ValueError(f'method `{method}` for qEpsilonMulti initialization is not implemented')
        return super().initialize(**kwargs)

    def _fixed_init(self, eps_alpha_dict, eps_beta_dict):
        for e in self.alpha_dict:
            self.alpha_dict[e] = eps_alpha_dict[e]
            self.beta_dict[e] = eps_beta_dict[e]

    def _uniform_init(self):
        # results in uniform (0,1)
        self._set_equal_params(1., 1.)

    def _non_mutation_init(self):
        self._set_equal_params(1., 10.)

    def _random_init(self, gamma_shape=2., gamma_rate=2., **kwargs):
        for e in self.alpha_dict.keys():
            a, b = torch.distributions.Gamma(gamma_shape, gamma_rate).sample((2,))
            self.alpha_dict[e] = a
            self.beta_dict[e] = b

    def create_masks(self, A):
        co_mut_mask = torch.zeros((A, A, A, A), dtype=torch.long)
        anti_sym_mask = torch.zeros((A, A, A, A), dtype=torch.long)
        absorbing_state_mask = torch.zeros((A, A, A, A), dtype=torch.long)
        # TODO: Find effecient way of indexing i-j = k-l
        for jj, j, ii, i in itertools.product(range(A), range(A), range(A), range(A)):
            if (ii == 0 and jj != 0) or (i == 0 and j != 0):
                absorbing_state_mask[jj, j, ii, i] = 1
            elif (jj - j) == (ii - i):
                co_mut_mask[jj, j, ii, i] = 1
            else:
                anti_sym_mask[jj, j, ii, i] = 1
        return co_mut_mask, anti_sym_mask, absorbing_state_mask

    def neg_cross_entropy(self, T_eval, w_T_eval):
        a = self.alpha_dict
        a_0 = self.alpha_prior
        b = self.beta_dict
        b_0 = self.beta_prior
        tot_CE = 0
        unique_edges, unique_edges_count = tree_utils.get_unique_edges(T_eval, self.config.n_nodes)
        H_edge = torch.zeros((self.config.n_nodes, self.config.n_nodes))
        for (u, v) in unique_edges:
            a_uv = a[u, v]
            a_uv_0 = a_0
            b_uv = b[u, v]
            b_uv_0 = b_0
            a_0_b_0_uv_tens = torch.tensor((a_uv_0, b_uv_0))
            log_Beta_ab_0_uv = math_utils.log_beta_function(a_0_b_0_uv_tens)
            psi_a_uv = torch.digamma(a_uv)
            psi_b_uv = torch.digamma(b_uv)
            psi_a_plus_b_uv = torch.digamma(a_uv + b_uv)

            H_edge[u, v] += torch.sum((log_Beta_ab_0_uv - (a_uv_0 - 1) * (psi_a_uv - psi_a_plus_b_uv) -
                                       (b_uv_0 - 1) * (psi_b_uv - psi_a_plus_b_uv)))

        for (T, w) in zip(T_eval, w_T_eval):
            for e in T.edges:
                tot_CE += H_edge[e] * w

        return -tot_CE

    def entropy(self, T_eval, w_T_eval):
        a = self.alpha_dict
        b = self.beta_dict
        tot_H = 0
        unique_edges, unique_edges_count = tree_utils.get_unique_edges(T_eval, self.config.n_nodes)
        H_edge = torch.zeros((self.config.n_nodes, self.config.n_nodes))
        for (u, v) in unique_edges:
            a_uv = a[u, v]
            b_uv = b[u, v]
            a_b_uv_tens = torch.tensor((a_uv, b_uv))
            log_Beta_ab_uv = math_utils.log_beta_function(a_b_uv_tens)
            psi_a_uv = torch.digamma(a_uv)
            psi_b_uv = torch.digamma(b_uv)
            psi_a_plus_b_uv = torch.digamma(a_uv + b_uv)
            H_edge[u, v] = torch.sum((log_Beta_ab_uv - (a_uv - 1) * psi_a_uv - (b_uv - 1) * psi_b_uv +
                                       (a_uv + b_uv - 2) * psi_a_plus_b_uv))

        for (T, w) in zip(T_eval, w_T_eval):
            for e in T.edges:
                tot_H += H_edge[e] * w
        return tot_H

    def compute_elbo(self, T_eval, w_T_eval) -> float:
        entropy_eps = self.entropy(T_eval, w_T_eval)
        CE_eps = self.neg_cross_entropy(T_eval, w_T_eval)
        return CE_eps + entropy_eps

    def h_eps0(self, i: Optional[int] = None, j: Optional[int] = None) -> Union[float, torch.Tensor]:
        if i is not None and j is not None:
            return 1. - self.config.eps0 if i == j else self.config.eps0 / (self.config.n_states - 1)
        else:
            heps0_arr = self.config.eps0 / (self.config.n_states - 1) * torch.ones(
                (self.config.n_states, self.config.n_states))
            diag_mask = get_zipping_mask0(self.config.n_states)
            heps0_arr[diag_mask] = 1 - self.config.eps0
            if i is None and j is not None:
                return heps0_arr[:, j]
            elif i is not None and j is None:
                return heps0_arr[i, :]
            else:
                return heps0_arr

    def exp_log_zipping(self, e: Tuple[int, int]) -> torch.Tensor:
        """Expected log-zipping function

        Parameters
        ----------
        e : Tuple[int, int]
            edge associated to the distance epsilon
        output : indexing [j', j, i', i]
        """
        u, v = e
        out_arr = torch.empty((self.config.n_states,) * 4)
        if self.fixed:
            # return the zipping function with true value of eps
            # which is the mean of the fixed distribution
            true_eps = self.true_params["eps"]
            try:
                out_arr[...] = torch.log(h_eps(self.config.n_states, true_eps[u, v]))
            except KeyError as ke:
                out_arr[...] = torch.log(h_eps(self.config.n_states, .8))  # distant clones if arc doesn't exist
        else:
            comut_mask, no_comut_mask, abs_state_mask = get_zipping_mask(self.config.n_states)
            A = normalizing_zipping_constant(self.config.n_states)
            digamma_a = torch.digamma(self.alpha_dict[u, v])
            digamma_b = torch.digamma(self.beta_dict[u, v])
            digamma_ab = torch.digamma(self.alpha_dict[u, v] + self.beta_dict[u, v])
            out_arr[...] = digamma_a - digamma_ab - A.log()
            out_arr[comut_mask] = digamma_b - digamma_ab
            out_arr[abs_state_mask] = -1000.
        return out_arr

    def exp_log_zipping_old(self, e: Tuple[int, int]) -> torch.Tensor:
        """Expected log-zipping function

        Parameters
        ----------
        e : Tuple[int, int]
            edge associated to the distance epsilon
        output : indexing [j', j, i', i]
        """
        u, v = e
        out_arr = torch.empty((self.config.n_states,) * 4)
        if self.fixed:
            # return the zipping function with true value of eps
            # which is the mean of the fixed distribution
            true_eps = self.true_params["eps"]
            try:
                out_arr[...] = torch.log(h_eps(self.config.n_states, true_eps[u, v]))
            except KeyError as ke:
                out_arr[...] = torch.log(h_eps(self.config.n_states, .8))  # distant clones if arc doesn't exist
        else:
            comut_mask = get_zipping_mask_old(self.config.n_states)
            A = normalizing_zipping_constant(self.config.n_states)
            digamma_a = torch.digamma(self.alpha_dict[u, v])
            digamma_b = torch.digamma(self.beta_dict[u, v])
            digamma_ab = torch.digamma(self.alpha_dict[u, v] + self.beta_dict[u, v])
            out_arr[...] = digamma_a - digamma_ab - A.log()
            out_arr[comut_mask] = digamma_b - digamma_ab
        return out_arr


    def mean(self) -> dict:
        mean_dict = {e: self.alpha_dict[e] / (self.alpha_dict[e] + self.beta_dict[e]) for e in self.alpha_dict.keys()}
        return mean_dict

    def var(self) -> dict:
        var_dict = {e: self.alpha_dict[e] * self.beta_dict[e] / ((self.alpha_dict[e] + self.beta_dict[e]) ** 2 *
                                                                 (self.alpha_dict[e] + self.beta_dict[e] + 1)) for e in
                    self.alpha_dict.keys()}
        return var_dict

    def __str__(self):
        # summary for qeps
        summary = ["[qEpsilon summary]"]
        if self.fixed:
            summary[0] += " - True Dist"
            summary.append(f"-eps\t\n{self.true_params['eps']}")  # prints dict
        else:
            # print top k smallest epsilons
            k = min(len(self.alpha_dict), 5)
            topk = sorted(self.mean().items(), key=lambda x: x[1])[:k]
            summary.append(f"-top{k}")
            var = self.var()
            for i in range(k):
                (u, v), e_mean = topk[i]
                e_var = var[u, v]
                summary.append(f"({u},{v}): {e_mean:.2f} " +
                               pm_uni +
                               f" {np.sqrt(e_var):.2f} (a={self.alpha_dict[u, v]:.2f}, b={self.beta_dict[u, v]:.2f})")

            summary.append(f"-prior\ta0={self.alpha_prior}, b0={self.beta_prior}")

        return os.linesep.join(summary)


# observations (mu-tau)
class qMuTau(qPsi):

    def __init__(self, config: Config, true_params=None,
                 nu_prior: float = 1., lambda_prior: float = .1,
                 alpha_prior: float = .5, beta_prior: float = .5):
        super().__init__(config, true_params is not None)

        # params for each cell
        self._nu = torch.empty(config.n_cells)
        self._lmbda = torch.empty(config.n_cells)
        self._alpha = torch.empty(config.n_cells)
        self._beta = torch.empty(config.n_cells)
        # prior / generative model
        self.nu_0 = torch.tensor(nu_prior)
        self.lmbda_0 = torch.tensor(lambda_prior)
        self.alpha_0 = torch.tensor(alpha_prior)
        self.beta_0 = torch.tensor(beta_prior)

        if true_params is not None:
            # for each cell, mean and precision of the emission model
            assert "mu" in true_params
            assert "tau" in true_params
        self.true_params = true_params

        self.params_history["nu"] = []
        self.params_history["lmbda"] = []
        self.params_history["alpha"] = []
        self.params_history["beta"] = []

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
        # logging.debug("- mu/tau updated")
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

    def initialize(self, method='fixed', **kwargs):
        """
        Args:
            method:
                -'fixed' [  loc: float = 1,
                            precision_factor: float = .1,
                            shape: float = 5,
                            rate: float = 5 ]
        Returns: self
        """
        if self.fixed:
            self._init_from_true_params()
        elif method == 'fixed':
            self._initialize_with_values(**kwargs)
        # elif method == 'clust-data':
        #     self._init_from_clustered_data(**kwargs)
        elif method == 'data':
            self._init_from_raw_data(**kwargs)
        else:
            raise ValueError(f'method `{method}` for qMuTau initialization is not implemented')

        return super().initialize(**kwargs)

    def _initialize_with_values(self, loc: float = 1, precision_factor: float = .1,
                                shape: float = 5, rate: float = 5, **kwargs):
        self.nu = loc * torch.ones(self.config.n_cells)
        self.lmbda = precision_factor * torch.ones(self.config.n_cells)
        self.alpha = shape * torch.ones(self.config.n_cells)
        self.beta = rate * torch.ones(self.config.n_cells)

    def _init_from_clustered_data(self, obs, clusters, copy_numbers):
        """
Initialize the mu and tau params to EM estimates given estimates of copy numbers and
clusters
        Args:
            obs: data, tensor (chain_length, n_cells)
            clusters: clustering of the data, tensor (n_cells,)
            copy_numbers: copy numbers estimate (e.g. Viterbi path) for each clone, tensor (n_nodes, chain_length)
        """
        # TODO: implement
        pass

    def _init_from_raw_data(self, obs: torch.Tensor):
        """
Initialize the mu and tau params given observations
        Args:
            obs: data, tensor (chain_length, n_cells)
        """
        # FIXME: test does not work
        self.nu = torch.mean(obs, dim=0)
        self.alpha = torch.ones((self.config.n_cells,))  # init alpha to small value (1)
        var = torch.var(obs, dim=0).clamp(min=.01)  # avoid 0 variance
        self.beta = var * self.alpha
        # set lambda to 1. (arbitrarily)
        self.lmbda = torch.tensor(1.) * torch.ones((self.config.n_cells,))

    def neg_cross_entropy_old(self) -> float:
        CE_prior = self.alpha_0 * torch.log(self.beta_0) + 0.5 * torch.log(self.lmbda_0) - torch.lgamma(self.alpha_0)
        CE_constants = 0.5 * torch.log(torch.tensor(2 * torch.pi))
        CE_var_terms = self.exp_log_tau()
        CE_cross_terms = - self.beta_0 * self.exp_tau() + (self.alpha_0 - 1) * self.exp_log_tau() - \
                         0.5 * self.lmbda_0 / self.lmbda
        CE_arr = CE_constants + CE_prior + CE_var_terms + CE_cross_terms
        return torch.sum(CE_arr)

    def entropy_old(self) -> float:
        entropy_prior = self.alpha * torch.log(self.beta) + 0.5 * torch.log(self.lmbda) - torch.lgamma(self.alpha)
        entropy_constants = 0.5 * torch.log(torch.tensor(2 * torch.pi))
        CE_var_terms = self.exp_log_tau()
        CE_cross_terms = self.beta * self.exp_tau() + (self.alpha - 1) * self.exp_log_tau() - \
                         0.5 * 1.  # lmbda / lmbda
        entropy_arr = entropy_constants + entropy_prior + CE_var_terms + CE_cross_terms
        return -torch.sum(entropy_arr)

    def entropy(self):
        ent = self.config.n_cells * .5 * np.log(2 * np.pi) + \
              .5 * (1 - self.beta.log() - self.lmbda.log()) + \
              (.5 - self.alpha) * torch.digamma(self.alpha) + self.alpha + torch.lgamma(self.alpha)

        if self.config.debug:
            assert ent.shape == (self.config.n_cells,)

        return torch.sum(ent)

    def neg_cross_entropy(self):
        neg_ce = - self.config.n_cells * .5 * np.log(2 * np.pi) + \
                 .5 * (self.lmbda_0.log() - self.lmbda_0 / self.lmbda) + \
                 (self.alpha_0 - .5) * (torch.digamma(self.alpha) - self.beta.log()) + \
                 self.alpha_0 * self.beta_0.log() - torch.lgamma(self.alpha_0) - self.beta_0 * self.alpha / self.beta

        if self.config.debug:
            assert neg_ce.shape == (self.config.n_cells,)

        return torch.sum(neg_ce)

    def compute_elbo(self) -> float:
        return self.neg_cross_entropy() + self.entropy()

    def elbo_old(self) -> float:
        return self.neg_cross_entropy_old() + self.entropy_old()

    def exp_log_emission(self, obs: torch.Tensor) -> torch.Tensor:
        M, N = obs.shape
        K = self.config.n_nodes
        A = self.config.n_states
        out_shape = (N, M, K, A)
        out_arr = torch.empty(out_shape)
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
            log_p_obs = true_dist.log_prob(obs[..., None])
            log_p_obs = log_p_obs[..., None].expand(M, N, A, K)
            out_arr[...] = torch.einsum('mniv->nmvi', log_p_obs)
        else:
            E_log_tau = self.exp_log_tau()
            E_tau = torch.einsum('mn,n->mn', torch.pow(obs, 2), self.exp_tau())
            E_mu_tau = torch.einsum('i,mn,n->imn', torch.arange(self.config.n_states), obs, self.exp_mu_tau())
            E_mu2_tau = torch.einsum('i,n->in', torch.pow(torch.arange(self.config.n_states), 2), self.exp_mu2_tau())[:,
                        None, :]
            log_p_obs = .5 * (E_log_tau - E_tau + 2. * E_mu_tau - E_mu2_tau)
            log_p_obs = log_p_obs.expand(K, A, M, N)
            out_arr[...] = torch.einsum('vimn->nmvi', log_p_obs)

        assert out_arr.shape == out_shape
        return out_arr

    def exp_tau(self):
        return self.alpha / self.beta

    def exp_log_tau(self):
        return torch.digamma(self.alpha) - torch.log(self.beta)

    def exp_mu_tau(self):
        return self.nu * self.alpha / self.beta

    def exp_mu2_tau(self):
        return 1. / self.lmbda + torch.pow(self.nu, 2) * self.alpha / self.beta

    def exp_mu2_tau_c(self):
        A = self.config.n_states
        N = self.config.n_cells
        c = torch.arange(0, A, dtype=torch.float)
        exp_c_lmbda = torch.einsum("i, n -> in", c, 1. / self.lmbda)
        exp_mu2_tau = torch.pow(self.nu, 2) * self.alpha / self.beta
        exp_sum = exp_c_lmbda + exp_mu2_tau
        return exp_sum

    def __str__(self):
        summary = ["[qMuTau summary]"]
        if self.fixed:
            summary[0] += " - True Dist"
            summary.append(f"-mu\t\t{self.true_params['mu'].mean(dim=-1):.2f} " +
                           pm_uni + f" {self.true_params['mu'].std(dim=-1):.2f}")
            summary.append(f"-tau\t{self.true_params['tau'].mean(dim=-1):.2f} " +
                           pm_uni + f" {self.true_params['tau'].std(dim=-1):.2f}")
        else:
            summary.append(f"-alpha\t{self.alpha.mean(dim=-1):.2f} " +
                           pm_uni + f" {self.alpha.std(dim=-1):.2f}" +
                           f" (prior {self.alpha_0:.2f})")
            summary.append(f"-beta\t{self.beta.mean(dim=-1):.2f} " +
                           pm_uni + f" {self.beta.std(dim=-1):.2f}" +
                           f" (prior {self.beta_0:.2f})")
            summary.append(f"-nu\t\t{self.nu.mean(dim=-1):.2f} " +
                           pm_uni + f" {self.nu.std(dim=-1):.2f}" +
                           f" (prior {self.nu_0:.2f})")
            summary.append(f"-lambda\t{self.lmbda.mean(dim=-1):.2f} " +
                           pm_uni + f" {self.lmbda.std(dim=-1):.2f}" +
                           f" (prior {self.lmbda_0:.2f})")
            summary.append(f"partial ELBO\t{self.compute_elbo():.2f}")

        return os.linesep.join(summary)

    def _init_from_true_params(self):
        self.alpha = self.alpha_0 + (self.config.chain_length + 1) * self.config.n_cells * .5
        self.beta = self.alpha / self.true_params['tau']
        self.nu = self.true_params['mu']
        self.lmbda = torch.ones(self.config.n_cells) * 100.


class qMuAndTauCellIndependent(VariationalDistribution):

    def __init__(self, config: Config, true_params=None):
        super().__init__(config, true_params is not None)

        # params for each cell
        self._nu = torch.empty(config.n_cells)
        self._phi = torch.empty(config.n_cells)
        self._alpha = torch.empty(1)
        self._beta = torch.empty(1)
        self.nu_0 = torch.empty(1)
        self.phi_0 = torch.empty(1)
        self.alpha_0 = torch.empty_like(self._alpha)
        self.beta_0 = torch.empty_like(self._beta)

        if true_params is not None:
            # for each cell, mean and precision of the emission model
            assert "mu" in true_params
            assert "tau" in true_params
        self.true_params = true_params

        self.params_history["nu"] = []
        self.params_history["phi"] = []
        self.params_history["alpha"] = []
        self.params_history["beta"] = []

    # getter ensures that params are only updated in
    # the class' update method
    @property
    def nu(self):
        return self._nu

    @property
    def phi(self):
        return self._phi

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @nu.setter
    def nu(self, n):
        self._nu[...] = n

    @phi.setter
    def phi(self, l):
        self._phi[...] = l

    @alpha.setter
    def alpha(self, a):
        self._alpha = a

    @beta.setter
    def beta(self, b):
        self._beta = b

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
        M = self.config.chain_length
        N = self.config.n_cells
        c_tensor = torch.arange(A, dtype=torch.float)
        q_Z = qz.exp_assignment()

        # tau update
        sum_MN_y2 = torch.pow(obs, 2).sum()  # sum over M and N
        E_mu = self.exp_mu()
        y_minus_Emu_c = obs.view(M, N, 1).expand(M, N, A) - torch.outer(E_mu, c_tensor).expand(M, N, A)
        sum_MCZ_y_minus_mu_c = torch.einsum("kma, nk, mna -> ", qc.single_filtering_probs, q_Z, y_minus_Emu_c ** 2)
        E_mu_n = self.exp_mu()
        Var_mu = self.phi
        E_mu_n_squared = E_mu_n ** 2 + Var_mu
        sum_MCZ_E_mu_squared = torch.einsum("kma, nk, a, n -> ", qc.single_filtering_probs, q_Z, c_tensor ** 2,
                                            E_mu_n_squared)
        A_tau = sum_MN_y2 - 2. * sum_MCZ_y_minus_mu_c + sum_MCZ_E_mu_squared
        B_tau = E_mu_n_squared.sum() - 2. * self.nu_0 * self.nu.sum() + N * self.nu_0 ** 2

        # mu update
        sum_MCZ_c2 = torch.einsum("kma, nk, a -> n", qc.single_filtering_probs, q_Z, c_tensor ** 2)
        sum_MCZ_cy = torch.einsum("kma, nk, a, mn -> n", qc.single_filtering_probs, q_Z, c_tensor, obs)
        E_tau = self.exp_tau()
        alpha = self.alpha_0 + (M + 1) * N * .5  # Never updated
        beta = self.beta_0 + .5 * (self.phi_0 * B_tau + A_tau)

        phi = E_tau * (self.phi_0 + sum_MCZ_c2)
        mu = (self.nu_0 * self.phi_0 + E_tau * sum_MCZ_cy) / phi

        # set new parameters
        new_mu, new_phi, new_alpha, new_beta = self.update_params(mu, phi, alpha, beta)

        super().update()
        return new_mu, new_phi, new_alpha, new_beta

    def update_params(self, mu, phi, alpha, beta):
        rho = self.config.step_size
        new_nu = (1 - rho) * self._nu + rho * mu
        new_phi = (1 - rho) * self._phi + rho * phi
        new_alpha = (1 - rho) * self._alpha + rho * alpha
        new_beta = (1 - rho) * self._beta + rho * beta
        self.nu = new_nu
        self.phi = new_phi
        self.alpha = new_alpha
        self.beta = new_beta
        return new_nu, new_phi, new_alpha, new_beta

    def initialize(self, loc: float = 10., precision_factor: float = 5.,
                   shape: float = 500., rate: float = 50., **kwargs):
        self.nu = loc * torch.ones(self.config.n_cells)
        self.phi = precision_factor * torch.ones(self.config.n_cells)
        self.alpha = torch.tensor(shape)
        self.beta = torch.tensor(rate)
        self.nu_0 = loc
        self.phi_0 = precision_factor
        self.alpha_0 = self._alpha
        self.beta_0 = self._beta
        return super().initialize(**kwargs)

    def cross_entropy(self) -> float:
        return super().compute_elbo()

    def entropy(self) -> float:
        return super().compute_elbo()

    def compute_elbo(self) -> float:
        return self.cross_entropy() + self.entropy()

    def exp_log_emission(self, obs: torch.Tensor) -> torch.Tensor:
        N, M, A = (self.config.n_cells, self.config.chain_length, self.config.n_states)
        out_shape = (N, M, A)
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
            E_tau = torch.einsum('mn, -> m', torch.pow(obs, 2), self.exp_tau())
            E_mu_tau = torch.einsum('i,mn,n -> imn', torch.arange(self.config.n_states), obs, self.exp_mu_tau())
            E_mu2_tau = torch.einsum('i,n -> in', torch.pow(torch.arange(self.config.n_states), 2), self.exp_mu2_tau())[
                        :, None, :]
            E_tau = E_tau.view(1, M, 1).expand(A, M, N)
            out_arr = .5 * (E_log_tau - E_tau + 2. * E_mu_tau - E_mu2_tau)
            out_arr = torch.einsum('imn->nmi', out_arr)

        assert out_arr.shape == out_shape
        return out_arr

    def exp_tau(self):
        return self.alpha / self.beta

    def exp_log_tau(self):
        return torch.digamma(self.alpha) - torch.log(self.beta)

    def exp_mu(self):
        return self.nu

    def exp_mu_tau(self):
        return self.nu * self.alpha / self.beta

    def exp_mu2_tau(self):
        return 1. / self.phi + torch.pow(self.nu, 2) * self.alpha / self.beta


class qTauUrn(VariationalDistribution):

    def __init__(self, config: Config, R: torch.Tensor, gc: torch.Tensor,
                 alpha_0: float = 50, beta_0: float = 10,
                 true_params=None):
        super().__init__(config, true_params is not None)

        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self._alpha = torch.empty(config.n_cells)
        self._beta = torch.empty(config.n_cells)
        self.R = R
        self.gc = gc
        self.phi = torch.empty(config.n_cells)
        if true_params is not None:
            # for each cell, mean and precision of the emission model
            assert "tau" in true_params
        self.true_params = true_params

        # NOTE: key-names of params_history must match the attributes
        #   so to be able to use `getattr(self, key_name)`
        self.params_history["alpha"] = []
        self.params_history["beta"] = []
        self.params_history["R"] = []
        self.params_history["gc"] = []
        self.params_history["phi"] = []

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

    def initialize(self, **kwargs):
        self.alpha = self.alpha_0
        self.beta = self.beta_0

    def update(self, qc: qC, qz: qZ, x: torch.Tensor):
        A = self.config.n_states
        M = self.config.chain_length
        N = self.config.n_cells
        R = self.R
        gc = self.gc
        self.update_phi(qc, qz)
        phi = self.phi
        c_tensor = torch.arange(A, dtype=torch.float)
        q_Z = qz.exp_assignment()
        q_C_marginals = qc.single_filtering_probs

        # tau update
        read_rates = torch.einsum("a, n, m -> anm", c_tensor, R / phi, gc)
        diff_term_squared = torch.pow(x.expand(A, N, M) - read_rates, 2)
        E_qZqC_diff = torch.einsum("nk, kma, anm -> n", q_Z, q_C_marginals, diff_term_squared)
        # mu update
        alpha = self.alpha_0 + M / 2  # Never updated
        beta = self.beta_0 + .5 * E_qZqC_diff

        # set new parameters
        new_alpha, new_beta = self.update_params(alpha, beta)

        super().update()
        return new_alpha, new_beta

    def update_params(self, alpha, beta):
        rho = self.config.step_size
        new_alpha = (1 - rho) * self._alpha + rho * alpha
        new_beta = (1 - rho) * self._beta + rho * beta
        self._alpha = new_alpha
        self._beta = new_beta
        return new_alpha, new_beta

    def update_phi(self, qc: qC, qz: qZ):
        qc_umj = qc.single_filtering_probs
        qz_nu = qz.pi
        j = torch.arange(self.config.n_states, dtype=torch.float)
        phi = torch.einsum("umj, nu, j -> n", qc_umj, qz_nu, j)
        self.phi = phi

    def cross_entropy(self) -> float:
        return super().compute_elbo()

    def entropy(self) -> float:
        return super().compute_elbo()

    def compute_elbo(self) -> float:
        return self.cross_entropy() + self.entropy()

    def exp_log_emission(self, obs: torch.Tensor) -> torch.Tensor:
        N, M, A = (self.config.n_cells, self.config.chain_length, self.config.n_states)
        out_shape = (N, M, A)
        out_arr = torch.ones(out_shape)
        R = self.R
        gamma = self.phi
        # obs is (m x n)
        if self.fixed:
            tau = self.true_params["tau"]

            # log emission is log normal with
            # mean=mu*cn_state, var=1/tau
            means = torch.outer(R / gamma,
                                torch.arange(self.config.n_states))
            true_dist = torch.distributions.Normal(loc=means,
                                                   scale=torch.ones(means.shape) / torch.sqrt(tau)[:, None])
            out_arr = torch.permute(true_dist.log_prob(obs[..., None]), (1, 0, 2))
        else:
            E_log_tau = self.exp_log_tau()
            E_tau = self.exp_tau()
            j = torch.arange(self.config.n_states)
            means = torch.einsum("j, n, m -> jnm", j, R / gamma, torch.ones((M,)))
            out_arr = .5 * (E_log_tau.expand(A, M, N) - torch.einsum("n, jnm -> jmn", E_tau,
                                                                     (obs.expand(A, N, M) - means) ** 2))
            out_arr = torch.einsum('jmn -> nmj', out_arr)

        assert out_arr.shape == out_shape
        return out_arr

    def exp_tau(self):
        return self._alpha / self._beta

    def exp_log_tau(self):
        return torch.digamma(self._alpha) - torch.log(self._beta)


class qTauRG(VariationalDistribution):

    def __init__(self, config: Config, R: torch.Tensor, alpha_0: float = 50, beta_0: float = 10, true_params=None):
        super().__init__(config, true_params is not None)

        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self._alpha = torch.empty(config.n_cells)
        self._beta = torch.empty(config.n_cells)
        self.R = R
        self.gamma = torch.empty(config.n_cells)
        if true_params is not None:
            # for each cell, mean and precision of the emission model
            assert "tau" in true_params
        self.true_params = true_params

        self.params_history["alpha"] = []
        self.params_history["beta"] = []
        self.params_history["R"] = []
        self.params_history["gamma"] = []

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

    def initialize(self, **kwargs):
        self.alpha = self.alpha_0
        self.beta = self.beta_0

    def update(self, qc: qC, qz: qZ, obs: torch.Tensor):
        A = self.config.n_states
        M = self.config.chain_length
        N = self.config.n_cells
        R = self.R
        self.update_gamma(qc, qz)
        gamma = self.gamma
        c_tensor = torch.arange(A, dtype=torch.float)
        q_Z = qz.exp_assignment()
        q_C_marginals = qc.single_filtering_probs

        # tau update
        read_rates = c_tensor.expand(N, A) * (R / gamma).expand(A, N).T
        read_rates = read_rates.expand(M, N, A)
        read_rates = torch.einsum("mna -> amn", read_rates)
        diff_term_squared = torch.pow(obs.expand(A, M, N) - read_rates, 2)
        E_qZqC_diff = torch.einsum("nk, kma, amn -> n", q_Z, q_C_marginals, diff_term_squared)
        # mu update
        alpha = self.alpha_0 + M / 2  # Never updated
        beta = self.beta_0 + .5 * E_qZqC_diff

        # set new parameters
        new_alpha, new_beta = self.update_params(alpha, beta)

        super().update()
        return new_alpha, new_beta

    def update_params(self, alpha, beta):
        rho = self.config.step_size
        new_alpha = (1 - rho) * self._alpha + rho * alpha
        new_beta = (1 - rho) * self._beta + rho * beta
        self._alpha = new_alpha
        self._beta = new_beta
        return new_alpha, new_beta

    def update_gamma(self, qc: qC, qz: qZ):
        qc_umj = qc.single_filtering_probs
        qz_nu = qz.pi
        j = torch.arange(self.config.n_states, dtype=torch.float)
        gamma = torch.einsum("umj, nu, j -> n", qc_umj, qz_nu, j)
        self.gamma = gamma

    def cross_entropy(self) -> float:
        return super().compute_elbo()

    def entropy(self) -> float:
        return super().compute_elbo()

    def compute_elbo(self) -> float:
        return self.cross_entropy() + self.entropy()

    def exp_log_emission(self, obs: torch.Tensor) -> torch.Tensor:
        N, M, A = (self.config.n_cells, self.config.chain_length, self.config.n_states)
        out_shape = (N, M, A)
        out_arr = torch.ones(out_shape)
        R = self.R
        gamma = self.gamma
        # obs is (m x n)
        if self.fixed:
            tau = self.true_params["tau"]

            # log emission is log normal with
            # mean=mu*cn_state, var=1/tau
            means = torch.outer(R / gamma,
                                torch.arange(self.config.n_states))
            true_dist = torch.distributions.Normal(loc=means,
                                                   scale=torch.ones(means.shape) / torch.sqrt(tau)[:, None])
            out_arr = torch.permute(true_dist.log_prob(obs[..., None]), (1, 0, 2))
        else:
            E_log_tau = self.exp_log_tau()
            E_tau = self.exp_tau()
            j = torch.arange(self.config.n_states)
            means = torch.outer(j, R / gamma).expand(M, A, N)
            means = torch.einsum("man->amn", means)
            out_arr = .5 * (E_log_tau - E_tau * (obs.expand(A, M, N) - means) ** 2)
            out_arr = torch.einsum('imn->nmi', out_arr)

        assert out_arr.shape == out_shape
        return out_arr

    def exp_tau(self):
        return self._alpha / self._beta

    def exp_log_tau(self):
        return torch.digamma(self._alpha) - torch.log(self._beta)


class qPhi(qPsi):

    def __init__(self, config: Config, phi_init, x, gc, R, A, emission_model="poisson", fixed=False):
        super().__init__(config, fixed)

        self.phi = phi_init
        self.x = x
        self.gc = gc
        self.R = R
        self.emission_model = emission_model

        self.params_history["phi"] = []
        self.params_history["x"] = []
        self.params_history["gc"] = []
        self.params_history["R"] = []

    def initialize(self, **kwargs):
        self.phi = torch.ones(self.config.n_nodes, ) * 2. * self.config.chain_length

    def update(self, qc: qC, qz: qZ, obs):
        c_marginals = qc.single_filtering_probs
        z_probs = qz.pi
        j = torch.arange(self.config.n_states, dtype=float)
        if self.emission_model.lower() == "poisson":
            phi = self.update_poisson(c_marginals, z_probs, j)
        self.phi = phi

    def update_poisson(self, c_marginals, z_probs, j):
        sum_means = torch.einsum("vmj, nv, m, n, j -> v", c_marginals, z_probs, self.gc, self.R.float(), j.float())
        sum_reads = torch.einsum("vmj, nv, nm -> v", c_marginals, z_probs, self.x)
        phi = sum_means / sum_reads
        return phi

    def update_lognormal(self):
        raise NotImplementedError

    def update_negbinomial(self):
        raise NotImplementedError

    def exp_log_emission(self, obs):
        out_shape = (self.config.n_cells, self.config.chain_length, self.config.n_nodes, self.config.n_states)
        out_arr = torch.ones(out_shape)
        # obs is (m x n)
        j = torch.arange(self.config.n_states)
        if self.emission_model.lower() == "poisson":
            out_arr = self.exp_log_emissions_poisson(j)

        assert out_arr.shape == out_shape
        return out_arr

    def exp_log_emissions_poisson(self, j):
        if self.fixed:
            phi = self.true_params["phi"]
        else:
            phi = self.phi

        K = self.config.n_nodes
        A = self.config.n_states
        N = self.config.n_cells
        M = self.config.chain_length
        rates = torch.einsum("j, m, n, v -> vjnm", j.float(), self.gc, self.R.float(), 1. / phi) + 0.00001
        poi_dist = torch.distributions.Poisson(rate=rates)
        out_arr = torch.permute(poi_dist.log_prob(self.x.expand(K, A, N, M)), (2, 3, 0, 1))
        assert out_arr.shape == (N, M, K, A)
        return out_arr


class qPi(VariationalDistribution):

    def __init__(self, config: Config, delta_prior: float = 1., true_params: dict | None = None):
        super().__init__(config, fixed=true_params is not None)

        self.concentration_param_prior = torch.ones(config.n_nodes) * delta_prior
        self._concentration_param = torch.empty_like(self.concentration_param_prior)

        if true_params is not None:
            assert "pi" in true_params
        self.true_params = true_params

        self.params_history["concentration_param"] = []

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
        # logging.debug("- pi updated")
        return new_concentration_param

    def exp_log_pi(self):
        e_log_pi = torch.empty_like(self.concentration_param)
        if self.fixed:
            e_log_pi[...] = torch.log(self.true_params["pi"])
        else:
            e_log_pi[...] = torch.digamma(self.concentration_param) - \
                            torch.digamma(torch.sum(self.concentration_param))

        return e_log_pi

    def exp_pi(self):
        e_pi = torch.empty_like(self.concentration_param)
        if self.fixed:
            e_pi[...] = self.true_params['pi']
        else:
            e_pi[...] = self.concentration_param / self.concentration_param.sum()
        return e_pi

    def neg_cross_entropy(self):
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

    def compute_elbo(self) -> float:
        cross_entropy = self.neg_cross_entropy()
        entropy = self.entropy()
        return cross_entropy + entropy

    def __str__(self):
        torch.set_printoptions(precision=3)
        summary = ["[qPi summary]"]
        if self.fixed:
            summary[0] += " - True Dist"
            summary.append(f"-pi\t{self.true_params['pi']}")
        else:
            summary.append(f"-delta\t{self.concentration_param}\n"
                           f"\t\t(prior: {self.concentration_param_prior})")
            summary.append(f"-E[pi]\t{self.exp_pi()}")

        return os.linesep.join(summary)
