class Config:
    def __init__(self,
            n_nodes: int = 10,
            n_states: int = 7,
            eps0: float = 1e-3,
            n_cells: int = 20,
            chain_length: int = 200,
            is_sample_size: int = 20) -> None:
        self._n_nodes = n_nodes
        self._n_states = n_states
        self._eps0 = eps0
        self._n_cells = n_cells
        self._chain_length = chain_length
        self._is_sample_size = is_sample_size # L in qT sampling

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
    def is_sample_size(self):
        return self._is_sample_size
        
