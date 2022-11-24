import torch
from utils.config import Config
from variational_distributions.q_Z import qZ
from variational_distributions.variational_distribution import VariationalDistribution
from variational_distributions.variational_hmm import CopyNumberHmm
from variational_distributions.variational_normal import qMuTau


class qPi(VariationalDistribution):

    def __init__(self, config: Config):
        self.concentration_param = torch.empty(config.n_nodes)
        super().__init__(config)

    def initialize(self):
        # initialize to balanced concentration (all ones)
        self.concentration_param = torch.ones(self.concentration_param.shape)
        return super().initialize()

    def update(self, qz: qZ, delta_pi_model):
        # pi_model = p(pi), parametrized by delta_k
        # generative model for pi

        self.concentration_param = delta_pi_model +\
            torch.sum(qz.exp_assignment(), dim=0)

        return super().update()

    def exp_log_pi(self):
        return torch.digamma(self.concentration_param) -\
                torch.digamma(torch.sum(self.concentration_param))

    def elbo(self) -> torch.Tensor:
        return super().elbo()
