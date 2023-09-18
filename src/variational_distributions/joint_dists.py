"""
File with classes of all joint variational distributions.
"""

from abc import abstractmethod
from typing import Union, List

import networkx as nx
import numpy as np
import torch
from numpy import infty

from utils import math_utils
from utils.config import Config
from utils.tree_utils import star_tree
from variational_distributions.observational_variational_distribution import qPsi
from variational_distributions.var_dists import qC, qZ, qT, qEpsilon, qEpsilonMulti, qMuTau, qPi, qPhi, qCMultiChrom
from variational_distributions.variational_distribution import VariationalDistribution


class JointDist(VariationalDistribution):
    def __init__(self, config: Config, fixed: bool = False):
        """
        Abstract class for collection of distributions which together give the joint.
        Parameters
        ----------
        config: Config, configuration object
        fixed: bool, True if parameters are fixed to true values and updates should not be performed
        """
        super().__init__(config, fixed)
        self._elbo: float = -infty

        self.params_history["elbo"] = []

    @property
    def elbo(self):
        return self._elbo

    @elbo.setter
    def elbo(self, e):
        self._elbo = e

    def get_params_as_dict(self) -> dict[str, np.ndarray]:
        return {
            'elbo': np.array(self.elbo)
        }

    def initialize(self, **kwargs):
        for q in self.get_units():
            q.initialize(**kwargs)
        return super().initialize(**kwargs)

    def update(self):
        # save elbo
        self.elbo = self.compute_elbo()
        super().update()

    def compute_elbo(self) -> float:
        return super().compute_elbo()

    @abstractmethod
    def get_units(self) -> List[VariationalDistribution]:
        return []

    def track_progress(self, reset=False):
        super().track_progress(reset)


class VarTreeJointDist(JointDist):
    def __init__(self, config: Config, obs: torch.Tensor,
                 qc=None, qz=None, qt=None, qeps=None, qmt=None, qpi=None):
        """
        Variational tree joint distribution class.
        """
        super().__init__(config)
        self.c: qC | qCMultiChrom = qC(config) if qc is None else qc
        self.z: qZ = qZ(config) if qz is None else qz
        self.t: qT = qT(config) if qt is None else qt
        self.eps: qEpsilon | qEpsilonMulti = \
            qEpsilonMulti(config, gedges=self.t.weighted_graph.edges) if qeps is None else qeps
        self.mt: qMuTau = qMuTau(config) if qmt is None else qmt
        self.pi: qPi = qPi(config) if qpi is None else qpi
        self.obs = obs

    @property
    def fixed(self):
        return all(q.fixed for q in self.get_units())

    @fixed.setter
    def fixed(self, f):
        pass

    def get_units(self) -> List[VariationalDistribution]:
        """
        Returns
        -------
        List of all variational distribution units.
        """
        # TODO: if needed, specify an ordering
        return [self.t, self.c, self.eps, self.pi, self.z, self.mt]

    def update(self, it=0):
        """
        Joint distribution update: update every variational unit in a predefined order.
        """
        self.t.update(self.c, self.eps)
        trees, weights = self.t.get_trees_sample()
        self.mt.update(self.c, self.z, self.obs)
        self.z.update(self.mt, self.c, self.pi, self.obs)
        self.c.update(self.obs, self.eps, self.z, self.mt, trees, weights)
        if self.config.qc_smoothing and it > int(self.config.n_run_iter / 10 * 6):
            self.c.smooth_etas()
            self.c.compute_filtering_probs()  #FIXME: calculated twice (here and in qC.update)
        self.eps.update(trees, weights, self.c)
        self.pi.update(self.z)


        super().update()

    def update_shuffle(self, n_updates: int = 5):
        """
        Joint distribution update: n_updates distributions in random order
        Parameters
        -------
        n_updates: int, number of random updates
        """
        self.t.update(self.c, self.eps)
        rnd_order = torch.randperm(n_updates)
        trees, weights = self.t.get_trees_sample()
        for i in range(n_updates):
            if rnd_order[i] == 0:
                self.mt.update(self.c, self.z, self.obs)
            elif rnd_order[i] == 1:
                self.c.update(self.obs, self.eps, self.z, self.mt, trees, weights)
            elif rnd_order[i] == 2:
                self.eps.update(trees, weights, self.c)
            elif rnd_order[i] == 3:
                self.pi.update(self.z)
            elif rnd_order[i] == 4:
                self.z.update(self.mt, self.c, self.pi, self.obs)


        return super().update()

    def initialize(self, **kwargs):
        return super().initialize(**kwargs)

    def compute_elbo(self, t_list: list | None = None, w_list: list | None = None) -> float:
        """
        Compute ELBO as sum of all distributions partial ELBOs. Trees sample can be provided as input (for
        larger samples) or created inside the function with pre-defined sample size.
        Parameters
        ----------
        t_list: list, trees in the provided sample
        w_list: list, weights of the trees in the sample

        Returns
        -------
        elbo, float
        """
        # FIXME: computation with part var, part fixed distributions is not implemented yet
        if t_list is None and w_list is None:
            t_list, w_list = self.t.get_trees_sample()

        elbo_tensor = self.c.compute_elbo(t_list, w_list, self.eps) + \
                      self.z.compute_elbo(self.pi) + \
                      self.mt.compute_elbo() + \
                      self.pi.compute_elbo() + \
                      self.eps.compute_elbo(t_list, w_list) + \
                      self.t.compute_elbo(t_list, w_list) + \
                      self.elbo_observations()
        return elbo_tensor.item()

    def elbo_observations(self):
        """
        Computes the partial ELBO for the observations. See formula in Supplementary Material.
        """
        E_log_tau = self.mt.exp_log_tau()
        E_tau = self.mt.exp_tau()
        E_mu_tau = self.mt.exp_mu_tau()
        E_mu2_tau = self.mt.exp_mu2_tau()

        qC = self.c.single_filtering_probs
        qZ = self.z.pi

        y = self.obs.detach().clone()
        nan_mask = torch.any(torch.isnan(y), dim=1)
        y[nan_mask, :] = 0.
        M_notnan = torch.sum(~nan_mask)
        M = self.config.chain_length
        A = self.config.n_states
        N = self.config.n_cells
        c = torch.arange(0, A, dtype=torch.float)
        c2 = c ** 2

        E_CZ_log_tau = torch.einsum("umi, nu, n, m ->", qC, qZ, E_log_tau, (~nan_mask).float()) if type(self.mt) is qMuTau\
            else torch.einsum(
            "umi, nu, m ->", qC, qZ, E_log_tau, (~nan_mask).float())  # TODO: possible to replace einsum with M * torch.sum(E_log_tau)?
        E_CZ_tau_y2 = torch.einsum("umi, nu, n, mn ->", qC, qZ, E_tau, y ** 2) if type(
            self.mt) is qMuTau else torch.einsum("umi, nu, , mn ->", qC, qZ, E_tau, y ** 2)
        E_CZ_mu_tau_cy = torch.einsum("umi, nu, n, mn, mni ->", qC, qZ, E_mu_tau, y, c.expand(M, N, A))
        E_CZ_mu2_tau_c2 = torch.einsum("umi, nu, n, i, m ->", qC, qZ, E_mu2_tau, c2, (~nan_mask).float())
        elbo = 1 / 2 * (E_CZ_log_tau - E_CZ_tau_y2 + 2 * E_CZ_mu_tau_cy - E_CZ_mu2_tau_c2 - N * M_notnan * torch.log(
            torch.tensor(2 * torch.pi)))
        if self.config.debug:
            assert not torch.isnan(elbo).any()
        return elbo

    def __str__(self):
        # summary for joint dist
        tot_str = "+++ Joint summary +++"
        tot_str += f"\n ELBO: {self.elbo}"
        for q in [self.t, self.c, self.eps, self.pi, self.z, self.mt]:
            tot_str += str(q)
            tot_str += "\n --- \n"
        tot_str += "+++ end of summary +++"
        return tot_str


class FixedTreeJointDist(JointDist):
    def __init__(self,
                 obs: torch.Tensor,
                 config: Config = None,
                 qc: qCMultiChrom | qC = None,
                 qz: qZ = None,
                 qeps: qEpsilonMulti | qEpsilon = None,
                 qpsi: qPsi = None,
                 qpi: qPi = None,
                 T: nx.DiGraph = None, R=None):
        """
        Fixed tree joint distribution. The topology is fixed in advance and passed as an input (T).
        Parameters
        -------
        T: networkx.DiGraph, tree topology
        """
        if config is None:
            config = Config(chain_length=obs.shape[0], n_cells=obs.shape[1])
        super().__init__(config)
        self.c: Union[qC, qCMultiChrom] = qc if qc is not None else qC(config)
        self.z: qZ = qz if qz is not None else qZ(config)
        self.eps: Union[qEpsilon, qEpsilonMulti] = qeps if qeps is not None else qEpsilonMulti(config)
        self.mt: qPsi = qpsi if qpsi is not None else qMuTau(config)
        self.pi: qPi = qpi if qpi is not None else qPi(config)
        self.obs = obs
        self.R = R
        # init to star tree if no tree is provided
        self.T = T if T is not None else star_tree(config.n_nodes)
        self.w_T = [1.0]

    def update(self, it=0):
        """
        Joint distribution update: update every variational unit in a predefined order.
        """
        self.c.update(self.obs, self.eps, self.z, self.mt, [self.T], self.w_T)
        if self.config.qc_smoothing and it > int(self.config.n_run_iter / 10 * 6):
            self.c.smooth_etas()
            self.c.compute_filtering_probs()
        self.eps.update([self.T], self.w_T, self.c)
        self.pi.update(self.z)
        self.z.update(self.mt, self.c, self.pi, self.obs)
        self.mt.update(self.c, self.z, self.obs)

        super().update()

    def SVI_update(self, it=0):
        """
        Joint distribution SVI update: update the local variables batch wise, then update global variables
        with smaller step size.
        """

        N = self.obs.shape[1]
        batches = torch.randperm(N)
        batch_size = self.config.batch_size
        n_batches = int(N / batch_size)
        for i in range(n_batches + 1):
            # Local updates
            if i == n_batches and N % batch_size != 0:
                batch = batches[i * batch_size: i * batch_size + N % batch_size]
            elif i == n_batches and N % batch_size == 0:
                continue
            else:
                batch = batches[i*batch_size: i*batch_size + batch_size]
            pi = self.z.update_CAVI(self.mt, self.c, self.pi, self.obs[:, batch], batch)
            self.z.update_params(pi, batch)
            mu, lmbda, alpha, beta = self.mt.update_CAVI(self.obs[:, batch], self.c, self.z, batch)
            self.mt.update_params(mu, lmbda, alpha, beta, batch)

            # Global updates
            new_eta1_norm, new_eta2_norm = self.c.update_CAVI(self.obs[:, batch], self.eps, self.z, self.mt, [self.T],
                                                              self.w_T, batch)
            self.c.update_params(new_eta1_norm, new_eta2_norm)
            self.c.compute_filtering_probs()

            if self.config.qc_smoothing and it > int(self.config.n_run_iter / 10 * 6):
                self.c.smooth_etas()
                self.c.compute_filtering_probs()
            alpha, beta = self.eps.update_CAVI([self.T], self.w_T, self.c)
            self.eps.update_params(alpha, beta)
            delta = self.pi.update_CAVI(self.z)
            self.pi.update_params(delta)

        # Call track progress only once per step
        if self.config.diagnostics:
            self.c.track_progress()
            self.z.track_progress()
            self.mt.track_progress()
            self.eps.track_progress()
            self.pi.track_progress()

        super().update()

    def get_units(self) -> list:
        # TODO: if needed, specify an ordering
        return [self.c, self.eps, self.pi, self.z, self.mt]

    @property
    def fixed(self):
        return all(q.fixed for q in self.get_units())

    @fixed.setter
    def fixed(self, f):
        pass

    def update_shuffle(self, n_updates: int = 5):
        """
        Joint distribution update: n_updates distributions in random order
        Parameters
        -------
        n_updates: int, number of random updates
        """
        rnd_order = torch.randperm(n_updates)
        for i in range(n_updates):
            if rnd_order[i] == 0:
                self.mt.update(self.c, self.z, self.obs)
            elif rnd_order[i] == 1:
                self.c.update(self.obs, self.eps, self.z, self.mt, [self.T], self.w_T)
            elif rnd_order[i] == 2:
                self.eps.update([self.T], self.w_T, self.c)
            elif rnd_order[i] == 3:
                self.pi.update(self.z)
            elif rnd_order[i] == 4:
                self.z.update(self.mt, self.c, self.pi, self.obs)

        return super().update()

    def initialize(self, **kwargs):
        return super().initialize(**kwargs)

    def compute_elbo(self) -> float:
        q_C_elbo = self.c.compute_elbo([self.T], self.w_T, self.eps)
        q_Z_elbo = self.z.compute_elbo(self.pi)
        q_MuTau_elbo = self.mt.compute_elbo()
        q_pi_elbo = self.pi.compute_elbo()
        q_eps_elbo = self.eps.compute_elbo([self.T], self.w_T)
        elbo_obs = self.elbo_observations()
        elbo_tensor = elbo_obs + q_C_elbo + q_Z_elbo + q_MuTau_elbo + q_pi_elbo + q_eps_elbo
        return elbo_tensor.item()

    def elbo_observations(self):
        """
        Computes the partial ELBO for the observations. See formula in Supplementary Material.
        """
        qC = self.c.single_filtering_probs
        qZ = self.z.pi
        A = self.config.n_states
        c = torch.arange(0, A, dtype=torch.float)

        if isinstance(self.mt, qMuTau):
            E_log_tau = self.mt.exp_log_tau()
            E_tau = self.mt.exp_tau()
            E_mu_tau = self.mt.exp_mu_tau()
            E_mu2_tau = self.mt.exp_mu2_tau()

            y = self.obs.detach().clone()
            nan_mask = torch.any(torch.isnan(y), dim=1)
            y[nan_mask, :] = 0.
            M_notnan = torch.sum(~nan_mask)

            c2 = c ** 2
            M, N = y.shape
            E_CZ_log_tau = torch.einsum("nu, n ->", qZ, E_log_tau) * M_notnan \
                if type(self.mt) is qMuTau \
                else torch.einsum("umi, nu, m ->", qC, qZ, E_log_tau, (~nan_mask).float())

            E_CZ_tau_y2 = torch.einsum("nu, n, mn ->", qZ, E_tau, y ** 2) if type(
                self.mt) is qMuTau else torch.einsum("umi, nu, , mn ->", qC, qZ, E_tau, y ** 2)
            E_CZ_mu_tau_cy = torch.einsum("umi, nu, n, mn, mni ->", qC, qZ, E_mu_tau, y, c.expand(M, N, A))
            E_CZ_mu2_tau_c2 = torch.einsum("umi, nu, n, i, m ->", qC, qZ, E_mu2_tau, c2, (~nan_mask).float())
            constant_term = N * M_notnan * torch.log(torch.tensor(2 * torch.pi))
            elbo = 1 / 2 * (E_CZ_log_tau - E_CZ_tau_y2 + 2 * E_CZ_mu_tau_cy - E_CZ_mu2_tau_c2 - constant_term)

            if self.config.debug:
                assert not torch.isnan(elbo).any()

        elif isinstance(self.mt, qPhi):
            # Poisson observational model
            x = self.obs
            R = self.mt.R
            gc = self.mt.gc
            phi = self.mt.phi
            if self.mt.emission_model.lower() == "poisson":
                lmbda = torch.einsum("nm, m, n, v, j -> jvnm", x, gc, R, 1./phi, c) + 0.00001
                log_x_factorial = math_utils.log_factorial(x)
                log_p = -log_x_factorial.expand(lmbda.shape) + torch.log(lmbda) + x.expand(lmbda.shape) - lmbda
                elbo = torch.einsum("vmj, nv, jvnm ->", qC, qZ, log_p)

        return elbo


class QuadrupletJointDist(JointDist):
    def __init__(self, config: Config,
                 qc, qz, qeps, qpsi, T: nx.DiGraph, obs: torch.Tensor, R=None):
        """
        Fixed tree joint distribution. The topology is fixed in advance and passed as an input (T).
        Parameters
        -------
        T: networkx.DiGraph, tree topology
        """
        super().__init__(config)
        self.c: qC = qc
        self.z: qZ = qz
        self.eps: Union[qEpsilon, qEpsilonMulti] = qeps
        self.mt: qPsi = qpsi
        self.obs = obs
        self.R = R
        self.T = T
        self.w_T = [1.0]

    def update(self, it=0):
        """
        Joint distribution update: update every variational unit in a predefined order.
        """
        self.mt.update(self.c, self.z, self.obs)
        self.c.update(self.obs, self.eps, self.z, self.mt, [self.T], self.w_T)
        if self.config.qc_smoothing and it > int(self.config.n_run_iter / 10 * 6):
            self.c.smooth_etas()
        self.c.compute_filtering_probs()
        self.eps.update([self.T], self.w_T, self.c)

        super().update()

    def get_units(self) -> list:
        # TODO: if needed, specify an ordering
        return [self.c, self.eps, self.pi, self.z, self.mt]

    def update_shuffle(self, n_updates: int = 3):
        """
        Joint distribution update: n_updates distributions in random order
        Parameters
        -------
        n_updates: int, number of random updates
        """
        rnd_order = torch.randperm(n_updates)
        for i in range(n_updates):
            if rnd_order[i] == 0:
                self.mt.update(self.c, self.z, self.obs)
            elif rnd_order[i] == 1:
                self.c.update(self.obs, self.eps, self.z, self.mt, [self.T], self.w_T)
            elif rnd_order[i] == 2:
                self.eps.update([self.T], self.w_T, self.c)

        return super().update()

    def initialize(self, **kwargs):
        return super().initialize(**kwargs)

    def compute_elbo(self) -> float:
        q_C_elbo = self.c.compute_elbo([self.T], self.w_T, self.eps)
        q_MuTau_elbo = self.mt.compute_elbo()
        q_eps_elbo = self.eps.compute_elbo([self.T], self.w_T)
        elbo_obs = self.elbo_observations()
        elbo_tensor = elbo_obs + q_C_elbo + q_MuTau_elbo + q_eps_elbo
        return elbo_tensor.item()

    def elbo_observations(self):
        """
        Computes the partial ELBO for the observations. See formula in Supplementary Material.
        """
        qC = self.c.single_filtering_probs
        qZ = self.z.pi
        A = self.config.n_states
        c = torch.arange(0, A, dtype=torch.float)

        if isinstance(self.mt, qMuTau):
            E_log_tau = self.mt.exp_log_tau()
            E_tau = self.mt.exp_tau()
            E_mu_tau = self.mt.exp_mu_tau()
            E_mu2_tau = self.mt.exp_mu2_tau()

            y = self.obs
            c2 = c ** 2
            M, N = y.shape
            E_CZ_log_tau = torch.einsum("umi, nu, n ->", qC, qZ, E_log_tau) if type(self.mt) is qMuTau else torch.einsum(
                "umi, nu, ->", qC, qZ, E_log_tau)  # TODO: possible to replace einsum with M * torch.sum(E_log_tau)?
            E_CZ_tau_y2 = torch.einsum("umi, nu, n, mn ->", qC, qZ, E_tau, y ** 2) if type(
                self.mt) is qMuTau else torch.einsum("umi, nu, , mn ->", qC, qZ, E_tau, y ** 2)
            E_CZ_mu_tau_cy = torch.einsum("umi, nu, n, mn, mni ->", qC, qZ, E_mu_tau, y, c.expand(M, N, A))
            E_CZ_mu2_tau_c2 = torch.einsum("umi, nu, n, i ->", qC, qZ, E_mu2_tau, c2)
            elbo = 1 / 2 * (E_CZ_log_tau - E_CZ_tau_y2 + 2 * E_CZ_mu_tau_cy - E_CZ_mu2_tau_c2 - N * M * torch.log(
                torch.tensor(2 * torch.pi)))

        elif isinstance(self.mt, qPhi):
            # Poisson observational model
            x = self.obs
            R = self.mt.R
            gc = self.mt.gc
            phi = self.mt.phi
            if self.mt.emission_model.lower() == "poisson":
                lmbda = torch.einsum("nm, m, n, v, j -> jvnm", x, gc, R, 1./phi, c) + 0.00001
                log_x_factorial = math_utils.log_factorial(x)
                log_p = -log_x_factorial.expand(lmbda.shape) + torch.log(lmbda) + x.expand(lmbda.shape) - lmbda
                elbo = torch.einsum("vmj, nv, jvnm ->", qC, qZ, log_p)

        return elbo