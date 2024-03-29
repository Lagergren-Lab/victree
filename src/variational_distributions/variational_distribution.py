"""
Interface class for Variational distributions
"""
import copy
import logging
from abc import abstractmethod

import numpy as np
import torch

from utils.config import Config


class VariationalDistribution:

    def __init__(self, config: Config, fixed: bool = False):
        self.config: Config = config
        self.fixed = fixed  # if true, don't update and exp-functions return fixed params
        self.params_history = {}

    def initialize(self, **kwargs):
        if self.config.diagnostics:
            self.track_progress(reset=True)
        return self

    def update(self):
        if self.config.diagnostics:
            self.track_progress()
        if self.fixed:
            logging.warning("Warn: fixed dist is updated!")

    def compute_elbo(self) -> float:
        return 0.

    def track_progress(self, reset=False):
        # TODO: add iteration number record
        # e.g.
        # self.params_history["it"].append(it)
        # NOTE: this function is never called by qCMultiChrom object
        if not self.config.diagnostics:
            raise Exception("progress tracking is being called but diagnostics are not requested")

        for k in self.params_history.keys():
            arr = getattr(self, k)
            if isinstance(arr, torch.Tensor):
                # deep-copy torch tensor with torch functions
                param_copy = arr.data.cpu().numpy().copy()
            elif isinstance(arr, np.ndarray):
                # copy an already existing numpy array
                param_copy = arr.copy()
            elif isinstance(arr, float) or isinstance(arr, int):
                # build 0-D np.ndarray (e.g. with elbo or iteration values)
                param_copy = np.array(arr, dtype=np.float32)
            else:
                # params not recognized, launch warning, but deep-copy anyway
                logging.warning(f"Saving non-array object to state checkpoint. Param {k} is of type {type(arr)}.")
                param_copy = copy.deepcopy(arr)
            if reset:
                self.params_history[k] = []
            self.params_history[k].append(param_copy)

    def reset_params_history(self):
        # empty params_history for any single qC dist
        for k in self.params_history.keys():
            self.params_history[k] = []

    @abstractmethod
    def get_params_as_dict(self):
        pass

    @abstractmethod
    def get_prior_params_as_dict(self):
        pass