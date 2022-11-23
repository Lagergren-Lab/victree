class Config:
    def __init__(self,
            n_nodes: int = 10,
            n_states: int = 7,
            eps0: float = 1e-3,
            n_cells: int = 20,
            chain_length: int = 200,
            wis_sample_size: int = 20,
            elbo_tol: float = 1e-10,
            max_close_runs: int = 10) -> None:
        self._n_nodes = n_nodes
        self._n_states = n_states
        self._eps0 = eps0
        self._n_cells = n_cells
        self._chain_length = chain_length
        self._wis_sample_size = wis_sample_size # L in qT sampling
        self._elbo_tol = elbo_tol
        self._max_close_runs = max_close_runs

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

    def __str__(self) -> str:
        s = f"config: K={self.n_nodes}," +\
            f"N={self.n_cells}," +\
            f"M={self.chain_length}," +\
            f"L={self.n_states}," +\
            f"cn_states={self.n_states}"

        return s
