import torch
from utils.config import Config
from variational_distributions.variational_distribution import VariationalDistribution


class qPi(VariationalDistribution):

    def __init__(self, config: Config, alpha_prior):
        super().__init__(config)
        self.concentration_param_prior = torch.ones(config.n_nodes) * alpha_prior
        self.concentration_param = self.concentration_param_prior

    def initialize(self):
        return super().initialize()

    def update(self, q_Z_probs: torch.Tensor):
        self.concentration_param = self.concentration_param_prior + torch.einsum("ij -> j", q_Z_probs)
        return self.concentration_param

    def exp_log_pi(self):
        delta_0 = torch.sum(self.concentration_param)
        return torch.digamma(self.concentration_param) - torch.digamma(delta_0)

    def differntial_entropy(self):
        dir_rv = torch.distributions.Dirichlet(self.concentration_param)
        return dir_rv.entropy()
