import torch
from utils.config import Config
from variational_distributions.variational_distribution import VariationalDistribution


class qZ(VariationalDistribution):
    def __init__(self, config: Config):
        super().__init__(config)

    # TODO: continue and implement exp_assignment
    def initialize(self, ):
        return super().initialize()

    def update(self):
        return super().update()

    def exp_assignment(self) -> torch.Tensor:
        # TODO: implement
        # expectation of assignment
        
        # temporarily uniform distribution
        return torch.ones((self.config.n_cells, self.config.n_nodes)) / self.config.n_nodes



