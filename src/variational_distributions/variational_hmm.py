from typing import Union
import networkx as nx
from networkx.convert_matrix import itertools
import torch
import torch.distributions as dist
from eps_utils import iter_pair_states

from variational_distributions.variational_distribution import VariationalDistribution


class CopyNumberHmm(VariationalDistribution):

    def __init__(self, n_nodes: int, chain_length: int, n_states: int):
        self.n_nodes = n_nodes
        self.chain_length = chain_length
        self.n_states = n_states

        self.single_filtering_probs = torch.empty((n_nodes, chain_length, n_states))
        self.couple_filtering_probs = torch.empty((n_nodes, chain_length, n_states, n_states))
        super().__init__()

    def update(self):
        self.update_CAVI()

    def update_CAVI(self, Y, T: nx.DiGraph, mu, sigma,):
        """
        log q*(C) += ( E_q(mu)q(sigma)[rho_Y(Y^u, mu, sigma)] + E_q(T)[E_{C^p_u}[eta(C^p_u, epsilon)] +
        + Sum_{u,v in T} E_{C^v}[rho_C(C^v,epsilon)]] ) dot T(C^u)

        CAVI update based on the dot product of the sufficient statistic of the HMM and simplified expected value over
        the natural parameter.
        :return:
        """
        n_nodes = T.number_of_nodes()
        T_C = torch.zeros((n_nodes, ))
        eta_C = torch.zeros((n_nodes, ))
        eta_C += self.E_mu_sigma_of_rho_Y(mu, sigma, Y)
        eta_C += self.E_mu_sigma_of_rho_Y(mu, sigma, Y)
        return torch.einsum()

    def E_mu_sigma_of_rho_Y(self, mu, sigma, Y, C):
        """
        Evaluates the
        :param mu: n_cells x 1 tensor
        :param sigma: n_cells x 1 tensor
        :param Y: n_cells x m_sites tensor
        :param C: n_nodes x m_sites tensor
        :return:
        """
        emission_dist = dist.Normal(mu*C, sigma)
        p_Y = emission_dist.log_prob(Y)
        E_mu_sigma_p_Y = torch.einsum('ij', 'ij-> ij', mu, p_Y)
        return E_mu_sigma_p_Y

    def exp_eta(self, tree: nx.DiGraph, 
            q_eps: VariationalDistribution,
            q_z: VariationalDistribution,
            q_mutau: VariationalDistribution) -> tuple[torch.Tensor, torch.Tensor]:
        """Expectation of natural parameter vector \eta

        Parameters
        ----------
        tree : nx.DiGraph
            Tree on which expectation is taken (e.g. sampled tree)
        q_eps : VariationalDistribution
            Variational distribution object of epsilon parameter
        q_z : VariationalDistribution
            Variational distribution object of cell assignment 
        q_mutau : VariationalDistribution
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

        e_eta1 = torch.zeros((self.n_nodes, self.chain_length, self.n_states))
        e_eta2 = torch.zeros((self.n_nodes, self.chain_length, self.n_states, self.n_states))

        # eta_1_iota(1, i)
        e_eta1[inner_nodes, 1, :] += torch.einsum('pj,ij->pi',
                self.single_filtering_probs[
                    [tree.predecessors(u)[0] for u in inner_nodes], 1, :],
                torch.tensor([q_eps.h_eps0(i, j) for (i, j) in iter_pair_states(self.n_states)]))
        # eta_1_iota(m, i)
        # TODO: define q_z and q_mutau functions
        e_eta1[inner_nodes, 1:, :] += torch.einsum('nv,nmi->vmi',
                q_z.assignment_probs,
                q_mutau.exp_log_emission())

        # eta_2_kappa(m, i, i')
        e_eta2[inner_nodes, :, :, :] += torch.einsum('pmjk,ihjk->pmih',
                self.couple_filtering_probs,
                q_eps.exp_zipping())

        # natural parameters for root node are fixed to healthy state
        e_eta1[root, 1, 2] = 1
        e_eta1[root, :, :, 2] = 1

        return e_eta1, e_eta2
            

    def exp_alpha(self, u, m, i: Union[int, tuple[int, int]], q_z = None, q_mutau = None):
        # alpha_iota(m, i)
        if isinstance(i, int):
            pass
        # alpha_kappa(m, i, i')
        else:
            pass

    def mc_filter(self, u, m, i: Union[int, tuple[int, int]]):
        if isinstance(i, int):
            return self.single_filtering_probs[u, m, i]
        else:
            return self.couple_filtering_probs[u, m, i[0], i[1]]

    # iota/kappa
    # might be useless
    def idx_map(self, m, i: Union[int, tuple[int, int]]) -> int:
        if isinstance(i, int):
            return m * self.n_states + i
        else:
            return m * (self.n_states ** 2) + i[0] * self.n_states + i[1]




