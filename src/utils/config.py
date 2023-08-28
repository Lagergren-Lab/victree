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
                 chromosome_indexes=None,
                 wis_sample_size: int = 5,
                 elbo_rtol: float = 5e-4,  # TODO: check if this is too low
                 max_close_runs: int = 10,
                 sieving_size: int = 1,
                 n_sieving_iter: int = 20,
                 step_size=1.,
                 annealing=1.0,
                 debug=False,
                 diagnostics=False,
                 out_dir="./output",
                 n_run_iter: int = 10,
                 save_progress_every_niter: int = 10,
                 qc_smoothing=False,
                 curr_it: int = 0,
                 split=False,
                 SVI=False,
                 batch_size=20) -> None:
        self.batch_size = batch_size
        self.SVI = SVI
        self.split = split
        self.curr_it = curr_it
        self.qc_smoothing = qc_smoothing
        self._diagnostics = diagnostics
        self.step_size = step_size
        self.annealing = annealing
        self._n_nodes = n_nodes
        self._n_states = n_states
        self._eps0 = eps0
        self._n_cells = n_cells
        self._chain_length = chain_length
        self._n_chromosomes = len(chromosome_indexes) + 1 if chromosome_indexes is not None else 1
        self._chromosome_indexes = chromosome_indexes
        self._wis_sample_size = wis_sample_size  # L in qT sampling
        self._elbo_rtol = elbo_rtol
        self._max_close_runs = max_close_runs
        self._sieving_size = sieving_size
        self._n_run_iter = n_run_iter
        self._n_sieving_iter = n_sieving_iter
        self._debug = debug
        self._out_dir = out_dir
        self._save_progress_every_niter = save_progress_every_niter

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

    @chain_length.setter
    def chain_length(self, chain_length):
        """
        Setter used in qCMultiChrom
        """
        self._chain_length = chain_length

    @property
    def n_chromosomes(self):
        return self._n_chromosomes

    @property
    def chromosome_indexes(self):
        return self._chromosome_indexes

    @chromosome_indexes.setter
    def chromosome_indexes(self, chr_idx: list):
        self._chromosome_indexes = chr_idx
        self._n_chromosomes = len(chr_idx) + 1

    @property
    def wis_sample_size(self):
        return self._wis_sample_size

    @property
    def elbo_rtol(self):
        return self._elbo_rtol

    @property
    def max_close_runs(self):
        return self._max_close_runs

    @property
    def sieving_size(self):
        return self._sieving_size

    @property
    def n_run_iter(self):
        return self._n_run_iter

    @property
    def n_sieving_iter(self):
        return self._n_sieving_iter

    @property
    def debug(self):
        return self._debug

    @property
    def diagnostics(self):
        return self._diagnostics

    @property
    def out_dir(self):
        return self._out_dir

    @property
    def save_progress_every_niter(self):
        return self._save_progress_every_niter

    @save_progress_every_niter.setter
    def save_progress_every_niter(self, save_progress_every_niter: int):
        self._save_progress_every_niter = save_progress_every_niter

    def __str__(self) -> str:
        s = f"config: K={self.n_nodes}," + \
            f"N={self.n_cells}," + \
            f"M={self.chain_length}," + \
            f"L={self.wis_sample_size}," + \
            f"cn_states={self.n_states}," + \
            f"n_chromosomes={self.n_chromosomes}," + \
            f"step_size={self.step_size}," + \
            f"sieving_size={self.sieving_size}"

        return s

    def to_dict(self):
        return self.__dict__


def set_seed(seed):
    # torch rng
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # python rng
    np.random.seed(seed)
    random.seed(seed)
