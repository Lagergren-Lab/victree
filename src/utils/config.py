import random

import numpy as np
import torch


class Config:
    def __init__(self,
                 n_nodes: int = 10,
                 n_states: int = 7,
                 eps0: float = 1e-2,
                 n_cells: int = 20,
                 chain_length: int = 200,
                 wis_sample_size: int = 5,
                 elbo_tol: float = 1e-10,
                 max_close_runs: int = 10,
                 sieving_size: int = 1,
                 n_sieving_iter: int = 20,
                 step_size=1.,
                 debug=False,
                 diagnostics=False) -> None:
        self._diagnostics = diagnostics
        self.step_size = step_size
        self._n_nodes = n_nodes
        self._n_states = n_states
        self._eps0 = eps0
        self._n_cells = n_cells
        self._chain_length = chain_length
        self._wis_sample_size = wis_sample_size  # L in qT sampling
        self._elbo_tol = elbo_tol
        self._max_close_runs = max_close_runs
        self._sieving_size = sieving_size
        self._n_sieving_iter = n_sieving_iter
        self._debug = debug

    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def n_states(self):
        return self._n_states

    @property
    def eps0(self):
        return self._eps0

    @property
    def n_cells(self):
        return self._n_cells

    @property
    def chain_length(self):
        return self._chain_length

    @property
    def wis_sample_size(self):
        return self._wis_sample_size

    @property
    def elbo_tol(self):
        return self._elbo_tol

    @property
    def max_close_runs(self):
        return self._max_close_runs

    @property
    def sieving_size(self):
        return self._sieving_size

    @property
    def n_sieving_iter(self):
        return self._n_sieving_iter

    @property
    def debug(self):
        return self._debug

    @property
    def diagnostics(self):
        return self._diagnostics

    def __str__(self) -> str:
        s = f"config: K={self.n_nodes}," + \
            f"N={self.n_cells}," + \
            f"M={self.chain_length}," + \
            f"L={self.wis_sample_size}," + \
            f"cn_states={self.n_states}," + \
            f"step_size={self.step_size}," + \
            f"sieving_size={self.sieving_size}"

        return s

    def to_dict(self):
        return {
            'n_nodes': self.n_nodes,
            'step_size': self.step_size,
            'n_states': self.n_states,
            'eps0': self.eps0,
            'n_cells': self.n_cells,
            'chain_length': self.chain_length,
            'wis_sample_size': self.wis_sample_size,
            'elbo_tol': self.elbo_tol,
            'max_close_runs': self.max_close_runs,
            'sieving_size': self.sieving_size,
            'n_sieving_runs': self.n_sieving_iter,
            'debug': self.debug
        }


def set_seed(seed):
    # torch rng
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # python rng
    np.random.seed(seed)
    random.seed(seed)
