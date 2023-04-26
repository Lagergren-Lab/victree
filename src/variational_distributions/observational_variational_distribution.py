"""
Interface class for Variational distributions
"""
import logging

from utils.config import Config
from variational_distributions.variational_distribution import VariationalDistribution


class qPsi(VariationalDistribution):

    def __init__(self, config: Config, fixed: bool = False):
        self.temp = 1.0
        super().__init__(config, fixed)

    def exp_log_emission(self):
        raise NotImplementedError

