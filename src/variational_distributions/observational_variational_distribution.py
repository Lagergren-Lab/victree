"""
Interface class for Variational distributions
"""
import torch

from utils.config import Config
from variational_distributions.variational_distribution import VariationalDistribution


class qPsi(VariationalDistribution):

    def __init__(self, config: Config, fixed: bool = False):
        self.temp = 1.0
        super().__init__(config, fixed)

    def exp_log_emission(self, obs: torch.Tensor):
        raise NotImplementedError

    def get_nan_mask(self, obs):
        _nan_mask = torch.zeros((obs.shape[1], obs.shape[0], self.config.n_nodes, self.config.n_states),
                                dtype=torch.bool)
        _nan_mask[torch.isnan(obs.T), ...] = True
        return _nan_mask
