from abc import abstractmethod
from typing import Union, List

import networkx as nx
import torch

from utils import math_utils
from utils.config import Config
from utils.tree_utils import tree_to_newick
from variational_distributions.observational_variational_distribution import qPsi
from variational_distributions.var_dists import qC, qZ, qT, qEpsilon, qEpsilonMulti, qMuTau, qPi, qPhi
from variational_distributions.variational_distribution import VariationalDistribution


class JointDist(VariationalDistribution):
    def __init__(self, config: Config, fixed: bool = False):
        super().__init__(config, fixed)
        self.diagnostics_dict = {} if config.diagnostics else None

    def initialize(self, **kwargs):
        for q in self.get_units():
            q.initialize(**kwargs)
        return super().initialize(**kwargs)

    def update(self):
        super().update()

    def elbo(self) -> float:
        return super().elbo()

    def init_diagnostics(self, n_iter: int):
        K, N, M, A = self.config.n_nodes, self.config.n_cells, self.config.chain_length, self.config.n_states
        # C, Z, pi diagnostics
        self.diagnostics_dict["C"] = torch.zeros((n_iter, K, M, A))
        self.diagnostics_dict["Z"] = torch.zeros((n_iter, N, K))
        self.diagnostics_dict["pi"] = torch.zeros((n_iter, K))

        # eps diagnostics
        self.diagnostics_dict["eps_a"] = torch.zeros((n_iter, K, K))
        self.diagnostics_dict["eps_b"] = torch.zeros((n_iter, K, K))

        # qMuTau diagnostics
        self.diagnostics_dict["nu"] = torch.zeros((n_iter, N))
        self.diagnostics_dict["lmbda"] = torch.zeros((n_iter, N))
        self.diagnostics_dict["alpha"] = torch.zeros((n_iter, N))
        self.diagnostics_dict["beta"] = torch.zeros((n_iter, N))

        self.diagnostics_dict["elbo"] = torch.zeros(n_iter)

    def update_diagnostics(self, iter: int):
        # C, Z, pi diagnostics
        self.diagnostics_dict["C"][iter] = self.c.single_filtering_probs
        self.diagnostics_dict["Z"][iter] = self.z.pi
        self.diagnostics_dict["pi"][iter] = self.pi.concentration_param

        # eps diagnostics
        K = self.config.n_nodes
        eps_a = torch.zeros((K, K))
        eps_b = torch.zeros((K, K))
        for key in self.eps.alpha.keys():
            eps_a[key] = self.eps.alpha[key]
            eps_b[key] = self.eps.beta[key]

        self.diagnostics_dict["eps_a"][iter] = eps_a
        self.diagnostics_dict["eps_b"][iter] = eps_b

        # qMuTau diagnostics
        self.diagnostics_dict["nu"][iter] = self.mt.nu
        self.diagnostics_dict["lmbda"][iter] = self.mt.lmbda
        self.diagnostics_dict["alpha"][iter] = self.mt.alpha  # not updated
        self.diagnostics_dict["beta"][iter] = self.mt.beta

        # elbo
        self.diagnostics_dict["elbo"][iter] = self.elbo()


    @abstractmethod
    def get_units(self) -> List[VariationalDistribution]:
        return []

    def track_progress(self, reset=False):
        # track all the variational dists in the joint model
        pass


class VarTreeJointDist(JointDist):
    def __init__(self, config: Config, obs: torch.Tensor,
                 qc=None, qz=None, qt=None, qeps=None, qmt=None, qpi=None):
        super().__init__(config)
        self.c: qC = qC(config) if qc is None else qc
        self.z: qZ = qZ(config) if qz is None else qz
        self.t: qT = qT(config) if qt is None else qt
        self.eps: qEpsilon | qEpsilonMulti = \
            qEpsilonMulti(config, gedges=self.t.weighted_graph.edges) if qeps is None else qeps
        self.mt: qMuTau = qMuTau(config) if qmt is None else qmt
        self.pi: qPi = qPi(config) if qpi is None else qpi
        self.obs = obs

    def get_units(self) -> List[VariationalDistribution]:
        # TODO: if needed, specify an ordering
        return [self.t, self.c, self.eps, self.pi, self.z, self.mt]

    def update(self):
        # T, C, eps, z, mt, pi
        self.t.update(self.c, self.eps)
        trees, weights = self.t.get_trees_sample()
        self.c.update(self.obs, self.eps, self.z, self.mt, trees, weights)
        self.eps.update(trees, weights, self.c)
        self.pi.update(self.z)
        self.z.update(self.mt, self.c, self.pi, self.obs)
        self.mt.update(self.c, self.z, self.obs)

    def initialize(self, **kwargs):
        return super().initialize(**kwargs)

    def elbo(self, t_list: list | None = None, w_list: list | None = None) -> float:
        if t_list is None and w_list is None:
            t_list, w_list = self.t.get_trees_sample()

        elbo_tensor = self.c.elbo(t_list, w_list, self.eps) + \
                      self.z.elbo(self.pi) + \
                      self.mt.elbo() + \
                      self.pi.elbo() + \
                      self.eps.elbo(t_list, w_list) + \
                      self.t.elbo(t_list, w_list) + \
                      self.elbo_observations()
        return elbo_tensor.item()

    def elbo_observations(self):
        E_log_tau = self.mt.exp_log_tau()
        E_tau = self.mt.exp_tau()
        E_mu_tau = self.mt.exp_mu_tau()
        E_mu2_tau = self.mt.exp_mu2_tau()

        qC = self.c.single_filtering_probs
        qZ = self.z.pi
        y = self.obs
        A = self.config.n_states
        c = torch.arange(0, A, dtype=torch.float)
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
        return elbo

    def __str__(self):
        # summary for joint dist
        tot_str = "+++ Joint summary +++"
        for q in [self.t, self.c, self.eps, self.pi, self.z, self.mt]:
            tot_str += str(q)
            tot_str += "\n --- \n"
        tot_str += "+++ end of summary +++"
        return tot_str

    # TODO: create a diagnostics class object

    def init_diagnostics(self, n_iter: int):
        super().init_diagnostics(n_iter)

        # add VarTree specific additional params
        L = self.config.wis_sample_size
        self.diagnostics_dict["wG"] = torch.zeros((n_iter, self.config.n_nodes, self.config.n_nodes))
        self.diagnostics_dict["wT"] = torch.zeros((n_iter, L))
        self.diagnostics_dict["gT"] = torch.zeros((n_iter, L))
        self.diagnostics_dict["T"] = []

    def update_diagnostics(self, iter: int):
        super().update_diagnostics(iter)

        self.diagnostics_dict["wG"][iter, ...] = torch.tensor(nx.to_numpy_array(self.t.weighted_graph))
        self.diagnostics_dict["wT"][iter] = self.t.w_T
        self.diagnostics_dict["gT"][iter] = self.t.g_T
        self.diagnostics_dict["T"].append([tree_to_newick(t) for t in self.t.get_trees_sample()[0]])


class FixedTreeJointDist(JointDist):
    def __init__(self, config: Config,
                 qc, qz, qeps, qpsi, qpi, T: nx.DiGraph, obs: torch.Tensor, R=None):
        super().__init__(config)
        self.c: qC = qc
        self.z: qZ = qz
        self.eps: Union[qEpsilon, qEpsilonMulti] = qeps
        self.mt: qPsi = qpsi
        self.pi: qPi = qpi
        self.obs = obs
        self.R = R
        self.T = T
        self.w_T = [1.0]

    def update(self):
        # T, C, eps, z, mt, pi
        self.mt.update(self.c, self.z, self.obs)
        self.c.update(self.obs, self.eps, self.z, self.mt, [self.T], self.w_T)
        self.eps.update([self.T], self.w_T, self.c)
        self.pi.update(self.z)
        self.z.update(self.mt, self.c, self.pi, self.obs)
        return super().update()

    def get_units(self) -> list:
        # TODO: if needed, specify an ordering
        return [self.c, self.eps, self.pi, self.z, self.mt]

    def update_shuffle(self):
        # T, C, eps, z, mt, pi
        n_updates = 5
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

    def elbo(self) -> float:
        q_C_elbo = self.c.elbo([self.T], self.w_T, self.eps)
        q_Z_elbo = self.z.elbo(self.pi)
        q_MuTau_elbo = self.mt.elbo()
        q_pi_elbo = self.pi.elbo()
        q_eps_elbo = self.eps.elbo([self.T], self.w_T)
        elbo_obs = self.elbo_observations()
        elbo_tensor = elbo_obs + q_C_elbo + q_Z_elbo + q_MuTau_elbo + q_pi_elbo + q_eps_elbo
        return elbo_tensor.item()

    def elbo_observations(self):
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

    def init_diagnostics(self, n_iter: int):
        super().init_diagnostics(n_iter)

    def update_diagnostics(self, iter: int):
        super().update_diagnostics(iter)
