"""
Interface class for Variational distributions
"""
import logging

from utils.config import Config


class VariationalDistribution:

    def __init__(self, config: Config, fixed: bool = False):
        self.config: Config = config
        self.fixed = fixed  # if true, don't update and exp-functions return fixed params

    def initialize(self, **kwargs):
        pass

    def update(self):
        if self.fixed:
            logging.warning("Warn: fixed dist is updated!")
        pass

    def elbo(self) -> float:
        return 0.
