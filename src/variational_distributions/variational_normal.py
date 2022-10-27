import torch
from utils.config import Config
from variational_distributions.variational_distribution import VariationalDistribution


class qMuTau(VariationalDistribution):

    def __init__(self, config: Config, loc: float, precision: float,
            shape: float, rate: float):
        # params for each cell
        self._loc = loc * torch.ones(config.n_cells)
        self._precision = precision * torch.ones(config.n_cells)
        self._shape = shape * torch.ones(config.n_cells)
        self._rate = rate * torch.ones(config.n_cells)
        super().__init__(config)

    # getter ensures that params are only updated in
    # the class' update method
    @property
    def loc(self):
        return self._loc

    @property
    def precision(self):
        return self._precision

    @property
    def shape(self):
        return self._shape

    @property
    def rate(self):
        return self._rate

    def update(self):
        return super().update()

    def initialize(self):
        return super().initialize()

    def exp_log_emission(self, obs: torch.Tensor) -> torch.Tensor:
        out_arr = torch.ones((self.config.n_cells, self.config.chain_length, self.config.n_states))
        
        out_arr = .5 * self.exp_log_tau() -\
                .5 * torch.einsum('mn,n->mn',
                        obs, self.exp_tau()) +\
                torch.einsum('i,mn,n->imn',
                        torch.arange(self.config.n_states),
                        obs,
                        self.exp_mu_tau()) -\
                .5 * torch.einsum('i,n->in',
                        torch.pow(torch.arange(self.config.n_states), 2),
                        self.exp_mu2_tau())

        return out_arr

    def exp_tau(self):
        return self.shape / self.rate

    def exp_log_tau(self):
        return torch.digamma(self.shape) - torch.log(self.rate)

    def exp_mu_tau(self):
        return self.loc * self.shape / self.rate

    def exp_mu2_tau(self):
        return 1. / self.precision +\
                torch.pow(self.loc, 2) * self.shape / self.rate


        



