import torch
from utils.config import Config
from variational_distributions.variational_distribution import VariationalDistribution


class qPi(VariationalDistribution):

    def __init__(self, config: Config):
        super().__init__(config)
        self.concentration_param = torch.ones(config.n_nodes) / config.n_nodes

    def initialize(self):
        return super().initialize()

    def update(self):
        return super().update()
