import torch
from utils.config import Config
from variational_distributions.variational_distribution import VariationalDistribution
import variational_distributions.q_pi as q_pi
import variational_distributions.variational_normal as variational_normal

class qZ(VariationalDistribution):
    def __init__(self, config: Config):
        self.K = config.n_nodes
        self.N = config.n_cells
        self.pi = self.initialize()  # N x K - categorical probability of cell n assigned to cluster K
        super().__init__(config)

    # TODO: continue and implement exp_assignment
    def initialize(self):
        return torch.ones((self.N, self.K)) / self.K  # init to equal probabilites

    def update(self, q_pi_dist: q_pi.qPi, q_mu_tau: variational_normal.qMuTau, q_C, obs: torch.Tensor):
        return self.update_CAVI(q_pi_dist, q_mu_tau, q_C, obs)

    def update_CAVI(self, q_pi_dist: q_pi.qPi, q_mu_tau: variational_normal.qMuTau, q_C, obs: torch.Tensor):
        """
        q(Z) is a Categorical with probabilities pi^{*}, where pi_k^{*} = exp(gamma_k) / sum_K exp(gamma_k)
        gamma_k = E[log \pi_k] + sum_{m,j} q(C_m^k = j) E_{\mu, \tau}[D_{nmj}]
        :param q_pi:
        :param q_mu_tau:
        :param q_C:
        :return:
        """
        E_log_pi = q_pi_dist.exp_log_pi()
        D = q_mu_tau.exp_log_emission(obs)
        gamma = E_log_pi + torch.einsum("mj, nmj -> n", q_C, D)
        self.pi = gamma / torch.exp(gamma)

    def exp_assignment(self) -> torch.Tensor:
        # TODO: implement
        # expectation of assignment
        
        # temporarily uniform distribution
        return torch.ones((self.config.n_cells, self.config.n_nodes)) / self.config.n_nodes



