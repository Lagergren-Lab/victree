from typing import Union
import networkx as nx
import torch
import torch.distributions as dist
from utils.config import Config
from variational_distributions.variational_distribution import VariationalDistribution
from variational_distributions.q_Z import qZ
from variational_distributions.q_epsilon import qEpsilon
from variational_distributions.q_T import q_T
from variational_distributions.variational_normal import qMuTau


class CopyNumberHmm(VariationalDistribution):

    def __init__(self, config: Config):

        super().__init__(config)
        self.single_filtering_probs = torch.empty((self.config.n_nodes, self.config.chain_length, self.config.n_states))
        self.couple_filtering_probs = torch.empty((self.config.n_nodes, self.config.chain_length, self.config.n_states, self.config.n_states))

        self.eta1 = torch.empty((self.config.n_nodes, self.config.chain_length, self.config.n_states))
        self.eta2 = torch.empty((self.config.n_nodes, self.config.chain_length, self.config.n_states, self.config.n_states))

    def initialize(self):
        # TODO: implement initialization of parameters
        return super().initialize()


    def log_density(self, copy_numbers: torch.Tensor, nodes: list = []) -> float:
        # compute probability of a copy number sequence over a set of nodes
        # if the nodes are not specified, whole q_c is evaluated (all nodes)
        # copy_numbers has shape (nodes, chain_length)
        # TODO: it's more complicated than expected, fixing it later
        if len(nodes):
            assert(copy_numbers.shape[0] == len(nodes))
            pass

        else:
            pass
        return 0.


    def update(self, obs: torch.Tensor, 
            q_t: q_T,
            q_eps: qEpsilon,
            q_z: qZ,
            q_mutau: qMuTau) -> tuple[torch.Tensor, torch.Tensor]:
        """
        log q*(C) += ( E_q(mu)q(sigma)[rho_Y(Y^u, mu, sigma)] + E_q(T)[E_{C^p_u}[eta(C^p_u, epsilon)] +
        + Sum_{u,v in T} E_{C^v}[rho_C(C^v,epsilon)]] ) dot T(C^u)

        CAVI update based on the dot product of the sufficient statistic of the 
        HMM and simplified expected value over the natural parameter.
        :return:
        """
        new_eta1 = torch.zeros((self.config.n_nodes, self.config.chain_length, self.config.n_states))
        new_eta2 = torch.zeros((self.config.n_nodes, self.config.chain_length, self.config.n_states, self.config.n_states)) 

        # FIXME: use weights for the importance sampling estimate of expectation
        for tree, weight in zip(*q_t.get_trees_sample()):
            # compute all alpha quantities
            exp_alpha1, exp_alpha2 = self.exp_alpha(q_eps)

            new_eta1, new_eta2 = self.exp_eta(obs, tree, q_eps, q_z, q_mutau)
            new_eta1 += torch.einsum('', exp_alpha1)
            for u in range(self.config.n_nodes):
                # for each node, get the children
                children = [w for w in tree.successors(u)]
                # sum on each node's update all children alphas
                new_eta1[u, :, :] += torch.einsum('wmi->mi', exp_alpha1[children, :, :])
                new_eta2[u, :, :, :] += torch.einsum('wmij->mij', exp_alpha2[children, :, :, :])
            
        return new_eta1, new_eta2

    def exp_eta(self, obs: torch.Tensor, tree: nx.DiGraph, 
            q_eps: qEpsilon,
            q_z: qZ,
            q_mutau: qMuTau) -> tuple[torch.Tensor, torch.Tensor]:
        """Expectation of natural parameter vector \eta

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

        e_eta1 = torch.zeros((self.config.n_nodes, self.config.chain_length, self.config.n_states))
        e_eta2 = torch.zeros((self.config.n_nodes, self.config.chain_length, self.config.n_states, self.config.n_states))

        # eta_1_iota(1, i)
        e_eta1[inner_nodes, 0, :] = torch.einsum('pj,ij->pi',
                self.single_filtering_probs[
                    [next(tree.predecessors(u)) for u in inner_nodes], 0, :],
                q_eps.h_eps0())
        # eta_1_iota(m, i)
        # TODO: define q_z and q_mutau functions
        e_eta1[inner_nodes, 1:, :] = torch.einsum('nv,nmi->vmi',
                q_z.exp_assignment(),
                q_mutau.exp_log_emission(obs))

        # eta_2_kappa(m, i, i')
        e_eta2[inner_nodes, :, :, :] = torch.einsum('pmjk,hikj->pmih',
                self.couple_filtering_probs,
                q_eps.exp_zipping())

        # natural parameters for root node are fixed to healthy state
        e_eta1[root, 0, 2] = 1
        e_eta1[root, :, :, 2] = 1

        return e_eta1, e_eta2
            

    def exp_alpha(self, q_eps: qEpsilon) -> tuple[torch.Tensor, torch.Tensor]:

        e_alpha1 = torch.zeros((self.config.n_nodes, self.config.chain_length, self.config.n_states))
        e_alpha2 = torch.zeros((self.config.n_nodes, self.config.chain_length, self.config.n_states, self.config.n_states))

        # alpha_iota(m, i)
        e_alpha1 = torch.einsum('wmj,ji->wmi',
                self.single_filtering_probs,
                # here q_eps.h_eps0 returns a tensor directly
                q_eps.h_eps0())

        # alpha_kappa(m, i, i')
        # similar to eta2 but with inverted indices in zipping
        e_alpha2 = torch.einsum('wmjk,kjhi->wmi',
                self.couple_filtering_probs,
                q_eps.exp_zipping())

        return e_alpha1, e_alpha2

    def mc_filter(self, u, m, i: Union[int, tuple[int, int]]):
        # TODO: implement/import forward-backward starting from eta params
        if isinstance(i, int):
            return self.single_filtering_probs[u, m, i]
        else:
            return self.couple_filtering_probs[u, m, i[0], i[1]]

    # iota/kappa
    # might be useless
    def idx_map(self, m, i: Union[int, tuple[int, int]]) -> int:
        if isinstance(i, int):
            return m * self.config.n_states + i
        else:
            return m * (self.config.n_states ** 2) + i[0] * self.config.n_states + i[1]




