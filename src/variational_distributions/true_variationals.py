from utils import math_utils
from utils.eps_utils import get_zipping_mask, get_zipping_mask0

import utils.tree_utils as tree_utils
from sampling.slantis_arborescence import sample_arborescence
from utils.config import Config
from variational_distributions.var_dists import *

# copy numbers
class qCTrue(qC):

    def __init__(self, config: Config, cn_profile: torch.Tensor):
        self.cn_profile = cn_profile

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
                                    q_eps.exp_zipping())
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
            q_C[u, :, :] = tree_utils.one_slice_marginals_markov_chain(self.eta1[u, 0, :], self.eta2[u],
                                                                       self.config.chain_length)

        return q_C

    def get_all_two_sliced_marginals(self):
        # TODO: optimize replacing for-loop with einsum operations
        q_C_pairs = torch.zeros(self.couple_filtering_probs.shape)
        for u in range(self.config.n_nodes):
            q_C_pairs[u, :, :, :] = tree_utils.two_slice_marginals_markov_chain(self.eta1[u, 0, :], self.eta2[u],
                                                                                self.config.chain_length)

        return q_C_pairs


# cell assignments
class qZTrue(qZ):
    def __init__(self, config: Config, cell_assignment: torch.Tensor):
        # cell assignment is a (n_cells,) tensor where for each
        # idx/cell = 0, ..., n_cells-1, the node is specified as int
        # the true pi is 1 where the cell is assigned to a node,
        # 0 otherwise
        self.pi = torch.zeros((config.n_cells, config.n_nodes))
        for c, u in enumerate(cell_assignment):
            self.pi[c, u] = 1.
        super().__init__(config)

    def exp_assignment(self) -> torch.Tensor:
        # simply the pi probabilities
        return self.pi

    def cross_entropy(self, qpi: 'qPi') -> float:
        e_logpi = qpi.exp_log_pi()
        return torch.einsum("nk, k -> ", self.pi, e_logpi)

    def entropy(self) -> float:
        return torch.special.entr(self.pi).sum()
        # return torch.einsum("nk, nk -> ", self.pi, torch.log(self.pi))

    def elbo(self, qpi: 'qPi') -> float:
        return self.cross_entropy(qpi) + self.entropy()


# topology
class qT(VariationalDistribution):

    def __init__(self, config: Config, fixed_tree: nx.DiGraph):
        super().__init__(config)
        self.fixed_tree = fixed_tree
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

    def get_trees_sample(self, alg="dslantis") -> Tuple[List, List]:
        # always return the same tree
        trees = [self.fixed_tree] * self.config.wis_sample_size
        weights = [1. / self.config.wis_sample_size] * self.config.wis_sample_size
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

    def __init__(self, config: Config, epsilon: torch.Tensor, alpha_0: float = 1., beta_0: float = 1.):
        self.epsilon = epsilon  # true values
        self.alpha_prior = torch.tensor(alpha_0)
        self.beta_prior = torch.tensor(beta_0)
        # one param for every arc except self referenced (diag set to -infty)
        self.alpha = torch.diag(-torch.ones(config.n_nodes) * np.infty) + alpha_0
        self.beta = torch.diag(-torch.ones(config.n_nodes) * np.infty) + beta_0
        super().__init__(config)

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

    def elbo(self) -> float:
        return super().elbo()

    def update(self, T_list, w_T, q_C_pairwise_marginals):
        self.update_CAVI(T_list, w_T, q_C_pairwise_marginals)

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
        copy_mask = get_zipping_mask(self.config.n_states)

        # FIXME: add normalization (division by A constant)
        norm_const = copy_mask * torch.sum(copy_mask, dim=(1, 2, 3))

        out_arr = torch.ones(copy_mask.shape) * \
                  (torch.digamma(self.beta[u, v]) - \
                   torch.digamma(self.alpha[u, v] + self.beta[u, v]))
        # select the combinations that do not satisfy i-i'=j-j'
        # and normalize
        out_arr[~copy_mask] -= norm_const
        return out_arr


# observations (mu-tau)
class qMuTau(VariationalDistribution):

    def __init__(self, config: Config, loc: float = 100, precision: float = .1,
                 shape: float = 5, rate: float = 5):
        # params for each cell
        self._loc = loc * torch.ones(config.n_cells)
        self._precision_factor = precision * torch.ones(config.n_cells)
        self._shape = shape * torch.ones(config.n_cells)
        self._rate = rate * torch.ones(config.n_cells)
        self.mu_prior = self._loc
        self.lambda_prior = self._precision_factor
        self.alpha_prior = self._shape
        self.alpha = self.alpha_prior + config.chain_length / 2  # alpha never updated
        self.beta_prior = self._rate
        super().__init__(config)

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
        out_arr = torch.ones((self.config.n_cells,
                              self.config.chain_length,
                              self.config.n_states))

        # FIXME: the output is not always of shape NxMxS
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

        return torch.einsum('imn->nmi', out_arr)

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
