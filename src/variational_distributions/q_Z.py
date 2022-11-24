import torch
from utils.config import Config
from variational_distributions.q_pi import qPi
from variational_distributions.variational_distribution import VariationalDistribution
from variational_distributions.variational_hmm import CopyNumberHmm
from variational_distributions.variational_normal import qMuTau


class qZ(VariationalDistribution):
    def __init__(self, config: Config):
        self.pi = torch.empty((config.n_cells, config.n_nodes))
        super().__init__(config)

    def initialize(self, ):
        # initialize to uniform
        self.pi = torch.ones((self.config.n_cells, self.config.n_nodes)) / self.config.n_nodes
        return super().initialize()

    def update(self, qmt: qMuTau, qc: CopyNumberHmm, qpi: qPi, obs: torch.Tensor):
        # single_filtering_probs: q(Cmk = j), shape: K x M x J
        qcmkj = qc.single_filtering_probs
        # expected log pi
        e_logpi = qpi.exp_log_pi()
        # Dnmj
        dnmj = qmt.exp_log_emission(obs)

        # op shapes: k + S_mS_j mkj nmj -> nk
        gamma = e_logpi + torch.einsum('mkj,nmj->nk', qcmkj, dnmj) 
        # TODO: remove asserts
        assert(gamma.shape == (self.config.n_cells, self.config.n_nodes))
        self.pi = torch.softmax(gamma, dim=1)
        assert(self.pi.shape == (self.config.n_cells, self.config.n_nodes))

        return super().update()

    def elbo(self) -> torch.Tensor:
        return super().elbo()

    def exp_assignment(self) -> torch.Tensor:
        # simply the pi probabilities
        return self.pi



