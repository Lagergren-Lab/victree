import logging

import hmmlearn.hmm
import torch
import torch.nn.functional as torch_functional
import networkx as nx
import itertools
import numpy as np
from typing import List, Tuple, Union, Optional

from sklearn.cluster import KMeans

from utils import math_utils
from utils.eps_utils import get_zipping_mask, get_zipping_mask0, h_eps, normalized_zipping_constant

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
                                      torch.ones((self.config.n_nodes, self.config.chain_length - 1)))

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
                                           implementation='scaling',
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
        qC_init = torch.distributions.Categorical(torch.exp(self.eta1))
        init_entropy = qC_init.entropy().sum()
        qC = torch.distributions.Categorical(torch.exp(self.eta2))
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

    def cross_entropy(self, T_list, w_T_list, q_eps: Union['qEpsilon', 'qEpsilonMulti']) -> float:
        # E_q[log p(C|...)]
        E_T = 0
        # v, m, j, j'
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
        # p(j' | j i' i)
        E_h_eps = q_eps.exp_log_zipping((u, v))
        cross_ent_pos_1 = torch.einsum("i,j,ji->",
                                       self.single_filtering_probs[u, 0, :],
                                       self.single_filtering_probs[v, 0, :],
                                       E_h_eps_0)
        cross_ent_pos_2_to_M = torch.einsum("mik, mjl, ljki->",
                                            # u, m, i, i'
                                            self.couple_filtering_probs[u, :, :, :],
                                            # v, m, j, j'
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
        logging.debug("- copy number updated")
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
                                              q_eps.h_eps0()) + \
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

    # obsolete
    def compute_fb_filtering_probs(self):
        # with forward-backward
        # TODO: remove
        self.single_filtering_probs = self._get_all_marginals()
        self.couple_filtering_probs = self._get_all_two_sliced_marginals()

    def _get_two_slice_marginals(self, u):
        return tree_utils.two_slice_marginals_markov_chain(self.eta1[u], self.eta2[u])

    def _get_marginals(self, u):
        return tree_utils.one_slice_marginals_markov_chain(self.eta1[u], self.eta2[u])

    def _get_all_marginals(self):
        q_C = torch.zeros(self.single_filtering_probs.shape)
        for u in range(self.config.n_nodes):
            init_eta = self.eta1[u, :]
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

    def _get_all_two_sliced_marginals(self):
        q_C_pairs = torch.zeros(self.couple_filtering_probs.shape)
        for u in range(self.config.n_nodes):
            init_eta = self.eta1[u, :]
            init_probs_qu = torch.exp(init_eta - torch.logsumexp(init_eta, dim=0))
            log_transition_probs = self.eta2[u]
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
        elif method == 'kmeans':
            self._kmeans_init(**kwargs)
        else:
            raise ValueError(f'method `{method}` for qZ initialization is not implemented')
        return super().initialize(**kwargs)

    def _random_init(self):
        # sample from a Dirichlet
        self.pi[...] = torch.distributions.Dirichlet(torch.ones_like(self.pi)).sample()

    def _uniform_init(self):
        # initialize to uniform probs among nodes
        self.pi[...] = torch.ones_like(self.pi) / self.config.n_nodes

    def _kmeans_init(self, obs, **kwargs):
        # TODO: find a soft k-means version
        # https://github.com/omadson/fuzzy-c-means
        # TODO: add normalization for observations
        eps = 1e-4
        m_obs = obs.mean(dim=0, keepdim=True)
        sd_obs = obs.std(dim=0, keepdim=True)
        # standardize to keep pattern
        scaled_obs = (obs - m_obs) / sd_obs.clamp(min=eps)
        kmeans = KMeans(n_clusters=self.config.n_nodes, random_state=0).fit(scaled_obs.T)
        m_labels = kmeans.labels_
        torch_labels = torch.tensor(m_labels)
        self.pi[...] = torch.nn.functional.one_hot(torch_labels.long(), num_classes=self.config.n_nodes)

    def _kmeans_per_site_init(self, obs, qmt: 'qMuTau'):
        M, N = obs.shape
        K = self.config.n_nodes
        A = self.config.n_states
        for m in range(M):
            kmeans = KMeans(n_clusters=K, random_state=0).fit(obs[m, :])
            m_labels = kmeans.labels_
        raise NotImplemented("kmeans_per_site_init not complete")

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
        d_nmj = qmt.exp_log_emission(obs)

        # op shapes: k + S_mS_j mkj nmj -> nk
        gamma = e_logpi + torch.einsum('kmj,nmj->nk', qc_kmj, d_nmj)
        pi = torch.softmax(gamma, dim=1)
        new_pi = self.update_params(pi)
        logging.debug("- z updated")
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
        return self.cross_entropy(qpi) - self.entropy()


# topology
class qT(VariationalDistribution):

    def __init__(self, config: Config, true_params=None):
        # weights are in log-form
        # so that tree.size() is log_prob of tree (sum of log_weights)
        self._weighted_graph = nx.DiGraph()
        self._weighted_graph.add_edges_from([(u, v)
                                             for u, v in itertools.permutations(range(config.n_nodes), 2)
                                             if u != v and v != 0])

        if true_params is not None:
            assert 'tree' in true_params
        self.true_params = true_params
        super().__init__(config, fixed=true_params is not None)

    @property
    def weighted_graph(self):
        return self._weighted_graph

    def initialize(self, **kwargs):
        # rooted graph with random weights in (0, 1) - log transformed
        self.init_fc_graph()
        return super().initialize(**kwargs)

    def cross_entropy(self):
        # sampled trees are not needed here
        # entropy = - log | T | (number of possible labeled directed rooted trees)
        # FIXME: cayleys formula is for undirected trees
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

    def elbo(self, trees, weights) -> float:
        """
Computes partial elbo for qT from the same trees-sample used for
other elbos such as qC.
        Args:
            trees: list of nx.DiGraph
            weights: list of weights as those in the qT.get_trees_sample() output
        Returns:
            float, value of ELBO for qT
        """
        # FIXME: gives weird values
        return self.cross_entropy() - self.entropy(trees, weights)

    def update(self, qc: qC, qeps: Union['qEpsilon', 'qEpsilonMulti']):
        # q_T = self.update_CAVI(T_list, qc, qeps)
        self.update_graph_weights(qc, qeps)
        logging.debug("- tree updated")
        return super().update()

    def update_params(self, new_weights: torch.Tensor):
        rho = self.config.step_size
        prev_weights = torch.tensor([w for u, v, w in self._weighted_graph.edges.data('weight')])
        stepped_weights = (1 - rho) * prev_weights + rho * new_weights

        # # minmax scaling the weights
        # stepped_weights -= stepped_weights.min()
        # stepped_weights /= stepped_weights.max()
        for i, (u, v, weight) in enumerate(self._weighted_graph.edges.data('weight')):
            self._weighted_graph.edges[u, v]['weight'] = stepped_weights[i]
        return self._weighted_graph.edges.data('weight')

    def update_graph_weights(self, qc: qC, qeps: Union['qEpsilon', 'qEpsilonMulti']):
        all_edges = [(u, v) for u, v in self._weighted_graph.edges]
        new_log_weights = {}
        for u, v in all_edges:
            new_log_weights[u, v] = torch.einsum('mij,mkl,jilk->', qc.couple_filtering_probs[u],
                                                 qc.couple_filtering_probs[v], qeps.exp_log_zipping((u, v)))
        # chain length determines how large are the log-weights
        # while they should be length invariant
        w_tensor = torch.tensor(list(new_log_weights.values())) / self.config.chain_length
        # # min-max scaling of weights
        # w_tensor -= torch.min(w_tensor)
        # w_tensor /= torch.max(w_tensor)
        self.update_params(w_tensor)
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

    def init_fc_graph(self):
        # random initialization of the fully connected graph over the clones
        for e in self._weighted_graph.edges:
            self._weighted_graph.edges[e]['weight'] = torch.rand(1).log()

    def get_trees_sample(self, alg: str = 'dslantis', sample_size: int = None) -> Tuple[List, List]:
        """
Sample trees from q(T) with importance sampling.
        Args:
            alg: string, chosen in ['random' | 'dslantis']
            sample_size: number of trees to be sampled. If None, sample_size is taken from the configuration
                object
        Returns:
            list of nx.DiGraph arborescences and list of related weights for computing expectations
            The weights are the result of the operation q'(T) / g'(T) where
                - q'(T) is the unnormalized probability under q(T), product of arc weights
                - g'(T) is the probability of the sample, product of Bernoulli trials (also unnormalized)
        """
        # e.g.:
        # trees = edmonds_tree_gen(self.config.is_sample_size)
        # trees = csmc_tree_gen(self.config.is_sample_size)
        trees = []
        l = self.config.wis_sample_size if sample_size is None else sample_size
        log_weights = torch.empty(l)
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
            for i in range(l):
                # t, w = sample_arborescence(log_W=log_W, root=0)
                t, log_isw = sample_arborescence_from_weighted_graph(self.weighted_graph)
                trees.append(t)
                log_q = t.size(weight='weight')  # unnormalized q(T)
                log_weights[i] = log_q - log_isw
        else:
            raise ValueError(f"alg '{alg}' is not implemented, check the documentation")

        weights = torch.exp(log_weights)
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
        logging.debug("- eps updated")
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

    def __init__(self, config: Config, alpha_0: float = 1., beta_0: float = 1., gedges=None,
                 true_params=None):
        # so that only admitted arcs are present (and self arcs such as v->v are not accessible)
        self.alpha_prior = torch.tensor(alpha_0)
        self.beta_prior = torch.tensor(beta_0)
        if gedges is None:
            # one param for every arc except self referenced and v -> root for any v
            gedges = [(u, v) for u, v in itertools.product(range(config.n_nodes),
                                                           range(config.n_nodes)) if v != 0 and u != v]
        self._alpha = {e: torch.tensor(self.alpha_prior) for e in gedges}
        self._beta = {e: torch.tensor(self.beta_prior) for e in gedges}

        if true_params is not None:
            assert "eps" in true_params
        self.true_params = true_params
        super().__init__(config, true_params is not None)

    def update_params(self, alpha: dict, beta: dict):
        rho = self.config.step_size
        for e in self.alpha.keys():
            self._alpha[e] = (1 - rho) * self.alpha[e] + rho * alpha[e]
            self._beta[e] = (1 - rho) * self.beta[e] + rho * beta[e]
        return self.alpha, self.beta

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, a: dict):
        for e, w in a.items():
            self._alpha[e] = w

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, b: dict):
        for e, w in b.items():
            self._beta[e] = w

    def set_all_equal_params(self, alpha: float, beta: float):
        for e in self.alpha.keys():
            self.alpha[e] = torch.tensor(alpha)
            self.beta[e] = torch.tensor(beta)

    def initialize(self, method='random', **kwargs):
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
        for e in self.alpha.keys():
            self.alpha[e] = torch.tensor(1.)
            self.beta[e] = torch.tensor(1.)

    def _random_init(self, gamma_shape=2., gamma_rate=2., **kwargs):
        a, b = torch.distributions.Gamma(gamma_shape, gamma_rate).sample((2,))
        for e in self.alpha.keys():
            self.alpha[e] = a
            self.beta[e] = b

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

    def cross_entropy(self, T_eval, w_T_eval):
        a = self.alpha
        a_0 = self.alpha_prior
        b = self.beta
        b_0 = self.beta_prior
        tot_H = 0
        unique_edges, unique_edges_count = tree_utils.get_unique_edges(T_eval, self.config.n_nodes)
        for (u, v), n_uv in zip(unique_edges, unique_edges_count):
            a_uv = a[u, v]
            a_uv_0 = a_0
            b_uv = b[u, v]
            b_uv_0 = b_0
            a_0_b_0_uv_tens = torch.tensor((a_uv_0, b_uv_0))
            Beta_ab_0_uv = math_utils.log_beta_function(a_0_b_0_uv_tens)
            psi_a_uv = torch.digamma(a_uv)
            psi_b_uv = torch.digamma(b_uv)
            psi_a_plus_b_uv = torch.digamma(a_uv + b_uv)
            tot_H += torch.sum(n_uv * (Beta_ab_0_uv - (a_uv_0 - 1) * (psi_a_uv - psi_a_plus_b_uv) -
                                       (b_uv_0 - 1) * (psi_b_uv - psi_a_plus_b_uv)))
        return tot_H

    def entropy(self, T_eval, w_T_eval):
        a = self.alpha
        b = self.beta
        tot_H = 0
        # TODO: replace n_uv by weights*n_uv
        unique_edges, unique_edges_count = tree_utils.get_unique_edges(T_eval, self.config.n_nodes)
        for (u, v), n_uv in zip(unique_edges, unique_edges_count):
            a_uv = a[u, v]
            b_uv = b[u, v]
            a_b_uv_tens = torch.tensor((a_uv, b_uv))
            Beta_ab_uv = math_utils.log_beta_function(a_b_uv_tens)
            psi_a_uv = torch.digamma(a_uv)
            psi_b_uv = torch.digamma(b_uv)
            psi_a_plus_b_uv = torch.digamma(a_uv + b_uv)
            tot_H -= torch.sum(n_uv * (Beta_ab_uv - (a_uv - 1) * psi_a_uv - (b_uv - 1) * psi_b_uv +
                                       (a_uv + b_uv + 2) * psi_a_plus_b_uv))
        return tot_H

    def elbo(self, T_eval, w_T_eval) -> float:
        entropy_eps = self.entropy(T_eval, w_T_eval)
        CE_eps = self.cross_entropy(T_eval, w_T_eval)
        return entropy_eps + CE_eps

    def update(self, tree_list: list, tree_weights: torch.Tensor, qc: qC):
        self.update_CAVI(tree_list, tree_weights, qc)
        super().update()

    def update_CAVI(self, tree_list: list, tree_weights: torch.Tensor, qc: qC):
        cfp = qc.couple_filtering_probs
        K, M, A, A = cfp.shape
        new_alpha = {(u, v): torch.tensor(self.alpha_prior) for u, v in self._alpha.keys()}
        new_beta = {(u, v): torch.tensor(self.beta_prior) for u, v in self._beta.keys()}
        # TODO: check how many edges are effectively updated
        #   after some iterations (might be very few)
        unique_edges, unique_edges_count = tree_utils.get_unique_edges(tree_list, N_nodes=K)

        # E_T[ sum_m sum_{not A} Cu Cv ]
        exp_cuv_a = {}
        # E_T[ sum_m sum_{A} Cu Cv ]
        exp_cuv_b = {}
        co_mut_mask, anti_sym_mask = self.create_masks(A)
        for u, v in unique_edges:
            exp_cuv_a[u, v] = torch.einsum('mij, mkl, ijkl -> ',
                                           cfp[u],
                                           cfp[v],
                                           anti_sym_mask)
            exp_cuv_b[u, v] = torch.einsum('mij, mkl, ijkl -> ',
                                           cfp[u],
                                           cfp[v],
                                           co_mut_mask)

        for k, t in enumerate(tree_list):
            for e in t.edges:
                ww = tree_weights[k]
                # only change the values related to the tree edges
                new_alpha[e] += ww * exp_cuv_a[e]
                new_beta[e] += ww * exp_cuv_b[e]

        self.update_params(new_alpha, new_beta)
        return new_alpha, new_beta

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

    def exp_log_zipping(self, e: Tuple[int, int]):
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
            # bool tensor with True on [j', j, i', i] where j'-j = i'-i (comutation)
            comut_mask = get_zipping_mask(self.config.n_states)

            # exp( E_CuCv[ log( 1 - eps) ] )
            # switching to exponential leads to easier normalization step
            # (same as the one in `h_eps()`)

            # FIXME: maybe exp( E[ log h ] ) needs not to be normalized.
            #   the update equation uses `- log A` instead, where A is the norm constant for each
            #   triplet (j, i', i)
            #   try to use `normalized_zipping_constant()` function
            # A = normalized_zipping_constant(self.config.n_states)
            beta_uv_tensor = torch.tensor(self.beta[u, v])
            alpha_uv_tensor = torch.tensor(self.alpha[u, v])
            exp_E_log_1meps = comut_mask * torch.exp(torch.digamma(beta_uv_tensor) -
                                                     torch.digamma(alpha_uv_tensor + beta_uv_tensor))
            exp_E_log_eps = (1. - exp_E_log_1meps.sum(dim=0)) / torch.sum(~comut_mask, dim=0)
            out_arr[...] = exp_E_log_eps * (~comut_mask) + exp_E_log_1meps
            if self.config.debug:
                assert torch.allclose(torch.sum(out_arr, dim=0), torch.ones_like(out_arr))
            out_arr[...] = out_arr.log()
        return out_arr

    def mean(self) -> dict:
        mean_dict = {e: self.alpha[e] / (self.alpha[e] + self.beta[e]) for e in self.alpha.keys()}
        return mean_dict


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
        logging.debug("- mu/tau updated")
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
        if method == 'fixed':
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
        self.nu_0[...] = self._nu
        self.lmbda_0[...] = self._lmbda
        self.alpha_0[...] = self._alpha
        self.beta_0[...] = self._beta

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
        self.alpha = torch.tensor(.5) * torch.ones((self.config.n_cells,))  # init alpha to small value
        var = torch.var(obs, dim=0).clamp(min=.01)  # avoid 0 variance
        self.beta = var * self.alpha
        # set lambda to 1. (arbitrarily)
        self.lmbda = torch.tensor(1.) * torch.ones((self.config.n_cells,))

    def cross_entropy(self) -> float:
        CE_prior = self.alpha_0 * torch.log(self.beta_0) + 0.5 * torch.log(self.lmbda_0) - torch.lgamma(self.alpha_0)
        CE_constants = 0.5 * torch.log(torch.tensor(2 * torch.pi))
        CE_var_terms = self.exp_log_tau()
        CE_cross_terms = - self.beta_0 * self.exp_tau() + (self.alpha_0 - 1) * self.exp_log_tau() - \
                         0.5 * self.lmbda_0 / self.lmbda
        CE_arr = CE_constants + CE_prior + CE_var_terms + CE_cross_terms
        return torch.sum(CE_arr)

    def entropy(self) -> float:
        entropy_prior = self.alpha * torch.log(self.beta) + 0.5 * torch.log(self.lmbda) - torch.lgamma(self.alpha)
        entropy_constants = 0.5 * torch.log(torch.tensor(2 * torch.pi))
        CE_var_terms = self.exp_log_tau()
        CE_cross_terms = self.beta * self.exp_tau() + (self.alpha - 1) * self.exp_log_tau() - \
                         0.5 * 1.  # lmbda / lmbda
        entropy_arr = entropy_constants + entropy_prior + CE_var_terms + CE_cross_terms
        return -torch.sum(entropy_arr)

    def entropy_alt(self):
        ent = - self.config.n_cells * .5 * np.log(2 * np.pi) + \
              .5 * (1 - self.beta.log() - self.lmbda.log()) + \
              (.5 - self.alpha) * torch.digamma(self.alpha) + self.alpha + torch.lgamma(self.alpha)

        if self.config.debug:
            assert ent.shape == (self.config.n_cells,)

        return torch.sum(ent)

    def neg_cross_entropy_alt(self):
        neg_ce = - self.config.n_cells * .5 * np.log(2 * np.pi) + \
                 .5 * (self.lmbda.log() - self.lmbda_0 / self.lmbda) + \
                 (self.alpha_0 - .5) * (torch.digamma(self.alpha) - self.beta.log()) + \
                 self.alpha_0 * self.beta_0.log() - torch.lgamma(self.alpha_0) - self.beta_0 * self.alpha / self.beta

        if self.config.debug:
            assert neg_ce.shape == (self.config.n_cells,)

        return torch.sum(neg_ce)

    def elbo(self) -> float:
        return self.cross_entropy() - self.entropy()

    def elbo_alt(self):
        return self.neg_cross_entropy_alt() + self.entropy_alt()

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
            E_mu2_tau = torch.einsum('i,n->in', torch.pow(torch.arange(self.config.n_states), 2), self.exp_mu2_tau())[:,
                        None, :]
            out_arr = .5 * (E_log_tau - E_tau + 2. * E_mu_tau - E_mu2_tau)
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
        logging.debug("- pi updated")
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
        return cross_entropy - entropy
