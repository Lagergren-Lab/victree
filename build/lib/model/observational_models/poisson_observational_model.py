import torch

from utils.config import Config
from variational_distributions.var_dists import qC, qZ

class PoissonObservationalModel():

    def __int__(self, config: Config, x: torch.Tensor, R:torch.Tensor, gc:torch.Tensor):
        self.x = x
        self.R = R
        self.gc = gc
        self.config = config

    def expected_log_emissions(self, q_psi):
        # Calculate E_{\Psi}[log p(D | Z, C, Psi)]
        # TODO: introduce qPsi class
        raise NotImplementedError

    def elbo_observations(self, qc: qC, qz: qZ, phi):
        """

        Parameters
        ----------
        self

        Returns
        -------

        """
        qC = qc.single_filtering_probs
        qZ = qz.z.pi
        x = self.x
        A = self.config.n_states
        c = torch.arange(0, A, dtype=torch.float)
        c2 = c ** 2
        M, N = x.shape
        j = torch.arange(self.config.n_states)
        means = torch.outer(j, self.R / phi).expand(M, A, N)
        means = torch.einsum("man->amn", means)
        out_arr = .5 * (x.expand(A, M, N) - means) ** 2
        out_arr = torch.einsum('imn->nmi', out_arr)
        raise NotImplementedError
