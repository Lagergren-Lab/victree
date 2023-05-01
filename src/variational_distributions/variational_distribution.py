"""
Interface class for Variational distributions
"""
import copy
import logging

import numpy as np
import torch

from utils.config import Config


class VariationalDistribution:

    def __init__(self, config: Config, fixed: bool = False):
        self.config: Config = config
        self.fixed = fixed  # if true, don't update and exp-functions return fixed params
        self.params_history: dict = {}

    def initialize(self, **kwargs):
        self.track_progress(reset=True)
        return self

    def update(self):
        if self.config.diagnostics:
            self.track_progress()
        if self.fixed:
            logging.warning("Warn: fixed dist is updated!")

    def elbo(self) -> float:
        return 0.

    def track_progress(self, reset=False):
        # TODO: add iteration number record
        # e.g.
        # self.params_history["it"].append(it)
        if not self.config.diagnostics:
            raise Exception("progress tracking is being called but diagnostics are not requested")

        for k in self.params_history.keys():
            arr = getattr(self, k)
            if isinstance(arr, torch.Tensor):
                param_copy = arr.data.cpu().numpy().copy()
            elif isinstance(arr, np.ndarray):
                param_copy = arr.copy()
            else:
                logging.warning(f"Saving non-array object to state checkpoint. Param {k} is of type {type(arr)}.")
                param_copy = copy.deepcopy(arr)
            if reset:
                self.params_history[k] = []
            self.params_history[k].append(param_copy)


