import networkx as nx
import torch
import torch.distributions as dist

from variational_distributions.variational_distribution import VariationalDistribution


class CopyNumberHmm(VariationalDistribution):

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

