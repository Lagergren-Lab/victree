"""
Interface class for Variational distributions
"""


import torch
from utils.config import Config


class VariationalDistribution:

    def __init__(self, config: Config):
        self.config: Config = config
        pass

    def initialize(self):
        pass

    def update(self):
        pass

    def elbo(self) -> torch.Tensor:
        return torch.empty(0)
