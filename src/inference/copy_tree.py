import copy
import logging
from typing import Union, List

import networkx as nx
from numpy import infty
import torch
import torch.distributions as dist

from utils.config import Config, set_seed
from variational_distributions.variational_distribution import VariationalDistribution
from variational_distributions.var_dists import qEpsilonMulti, qT, qEpsilon, qMuTau, qPi, qZ, qC, \
    qMuAndTauCellIndependent


class JointVarDist(VariationalDistribution):
    def __init__(self, config: Config, obs: torch.Tensor,
                 qc=None, qz=None, qt=None, qeps=None, qmt=None, qpi=None):
        super().__init__(config)
        self.c: qC = qC(config) if qc is None else qc
        self.z: qZ = qZ(config) if qz is None else qz
        self.t: qT = qT(config) if qt is None else qt
        self.eps: Union[qEpsilon, qEpsilonMulti] = \
            qEpsilonMulti(config, gedges=self.t.weighted_graph.edges) if qeps is None else qeps
        self.mt: qMuTau = qMuTau(config) if qmt is None else qmt
        self.pi: qPi = qPi(config) if qpi is None else qpi
        self.obs = obs

    def update(self):
        # T, C, eps, z, mt, pi
        self.t.update(self.c, self.eps)
        trees, weights = self.t.get_trees_sample()
        self.c.update(self.obs, self.eps, self.z, self.mt, trees, weights)
        self.eps.update(trees, weights, self.c)
        self.pi.update(self.z)
        self.z.update(self.mt, self.c, self.pi, self.obs)
        self.mt.update(self.c, self.z, self.obs)

        return super().update()

    def initialize(self, **kwargs):
        for q in [self.t, self.c, self.eps, self.pi, self.z, self.mt]:
            q.initialize(**kwargs)
        return super().initialize()

    def elbo(self, T_eval, w_T_eval) -> float:
        return self.c.elbo(T_eval, w_T_eval, self.eps) + \
               self.z.elbo(self.pi) + \
               self.mt.elbo() + \
               self.pi.elbo() + \
               self.eps.elbo(T_eval, w_T_eval) + \
               self.t.elbo(T_eval, w_T_eval) + \
               self.elbo_observations()

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


class VarDistFixedTree(VariationalDistribution):
    def __init__(self, config: Config,
                 qc, qz, qeps, qmt, qpi, T: nx.DiGraph, obs: torch.Tensor, R=None):
        super().__init__(config)
        self.c: qC = qc
        self.z: qZ = qz
        self.eps: Union[qEpsilon, qEpsilonMulti] = qeps
        self.mt: Union[qMuTau, qMuAndTauCellIndependent] = qmt
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
        for q in [self.c, self.eps, self.pi, self.z, self.mt]:
            q.initialize(**kwargs)

        return super().initialize()

    def elbo(self) -> float:
        q_C_elbo = self.c.elbo([self.T], self.w_T, self.eps)
        q_Z_elbo = self.z.elbo(self.pi)
        q_MuTau_elbo = self.mt.elbo()
        q_pi_elbo = self.pi.elbo()
        q_eps_elbo = self.eps.elbo([self.T], self.w_T)
        elbo_obs = self.elbo_observations()
        return elbo_obs + q_C_elbo + q_Z_elbo + q_MuTau_elbo + q_pi_elbo + q_eps_elbo

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


class CopyTree:

    def __init__(self, config: Config,
                 q: Union[JointVarDist, VarDistFixedTree],
                 obs: torch.Tensor):

        self.config = config
        self.q = q
        self.obs = obs
        self.diagnostics_dict = {} if config.diagnostics else None

        # counts the number of steps performed
        self.it_counter = 0
        self.elbo: float = -infty
        self.sieve_models: List[Union[JointVarDist, VarDistFixedTree]] = []

    def run(self, n_iter):

        # counts the number of irrelevant updates
        close_runs = 0

        if self.diagnostics_dict is not None:
            self.init_diagnostics(n_iter + 1)  # n_iter + 1 for initialization values
            self.update_diagnostics(0)

        if self.config.sieving_size > 1:
            self.compute_elbo()
            logging.info(f"ELBO before sieving: {self.elbo:.2f}")
            self.sieve(self.config.n_sieving_runs)
            self.compute_elbo()
            logging.info(f"ELBO after sieving: {self.elbo:.2f}")
        else:
            self.compute_elbo()
            logging.info(f"ELBO after init: {self.elbo:.2f}")

        logging.info("Start updates...")
        for it in range(1, n_iter + 1):
            # do the updates
            if it % 10 == 0:
                logging.info(f"It: {it}")
            else:
                logging.debug(f"It: {it}")
            self.step()

            old_elbo = self.elbo
            self.compute_elbo()
            if it % 10 == 0:
                logging.info(f"[{it}] ELBO: {self.elbo:.2f}")

            if self.diagnostics_dict is not None:
                self.update_diagnostics(it)

            if abs(old_elbo - self.elbo) < self.config.elbo_tol:
                close_runs += 1
                if close_runs > self.config.max_close_runs:
                    logging.debug(f"Run ended after {it}/{n_iter} iterations due to plateau")
                    #break
            elif self.elbo < old_elbo:
                # elbo should only increase
                # raise ValueError("Elbo is decreasing")
                logging.debug("Elbo is decreasing")
            else:
                close_runs = 0

        print(f"ELBO final: {self.elbo:.2f}")

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

        if type(self.q) is JointVarDist:
            L = self.config.wis_sample_size
            self.diagnostics_dict["wG"] = torch.zeros((n_iter, K, K))
            self.diagnostics_dict["wT"] = torch.zeros((n_iter, L))
            self.diagnostics_dict["gT"] = torch.zeros((n_iter, L))
            self.diagnostics_dict["T"] = []

    def update_diagnostics(self, iter: int):
        # C, Z, pi diagnostics
        self.diagnostics_dict["C"][iter] = self.q.c.single_filtering_probs
        self.diagnostics_dict["Z"][iter] = self.q.z.pi
        self.diagnostics_dict["pi"][iter] = self.q.pi.concentration_param

        # eps diagnostics
        K = self.config.n_nodes
        eps_a = torch.zeros((K, K))
        eps_b = torch.zeros((K, K))
        for key in self.q.eps.alpha.keys():
            eps_a[key] = self.q.eps.alpha[key]
            eps_b[key] = self.q.eps.beta[key]

        self.diagnostics_dict["eps_a"][iter] = eps_a
        self.diagnostics_dict["eps_b"][iter] = eps_b

        # qMuTau diagnostics
        self.diagnostics_dict["nu"][iter] = self.q.mt.nu
        self.diagnostics_dict["lmbda"][iter] = self.q.mt.lmbda
        self.diagnostics_dict["alpha"][iter] = self.q.mt.alpha  # not updated
        self.diagnostics_dict["beta"][iter] = self.q.mt.beta

        # elbo
        self.diagnostics_dict["elbo"][iter] = self.elbo

        if type(self.q) is JointVarDist:
            self.diagnostics_dict["wG"][iter, ...] = torch.tensor(nx.to_numpy_array(self.q.t.weighted_graph))
            self.diagnostics_dict["wT"][iter] = self.q.t.w_T
            self.diagnostics_dict["gT"][iter] = self.q.t.g_T
            self.diagnostics_dict["T"].append(self.q.t.T_list)


    def sieve(self, n_sieve_iter=10, seed_list=None):
        """
        Creates self.config.sieving_size number of copies of self.q, re-initializes each q with different
        seeds, performs n_sieve_iter updates of q, calculates the ELBO of each copy and sets self.q to the best
        copy with largest ELBO.
        :param n_sieve_iter: number of updates before sieving selection
        :return:
        """
        seed_list = range(self.config.sieving_size) if seed_list is None else seed_list
        for i in range(self.config.sieving_size):
            self.sieve_models.append(copy.deepcopy(self.q))
            set_seed(seed_list[i])
            self.sieve_models[i].initialize()

            for j in range(n_sieve_iter):
                self.sieve_models[i].update()

        selected_model_idx = self.sieving_selection_ELBO()
        logging.info(f"Selected sieve model index: {selected_model_idx} with seed: {seed_list[selected_model_idx]}")
        self.q = self.sieve_models[selected_model_idx]

    def sieving_selection_ELBO(self):
        elbos = []
        for i in range(self.config.sieving_size):
            elbos.append(self.sieve_models[i].elbo())

        logging.info(f"Sieved elbos: {elbos}")
        max_elbo_idx = torch.argmax(torch.tensor(elbos))
        return max_elbo_idx

    def sieving_selection_likelihood(self):
        log_L = []

        for i in range(self.config.sieving_size):
            q_i = self.sieve_models[i]
            qC_marginals = q_i.c.single_filtering_probs
            max_prob_cat = torch.argmax(qC_marginals, dim=-1)
            exp_var_mu = q_i.mt.nu
            exp_var_tau = q_i.mt.exp_tau()
            log_L_var_model = 0
            for n in range(self.config.n_cells):
                y_n = self.obs[:, n]
                u_var = torch.argmax(q_i.z.pi[n])
                obs_model_var = dist.Normal(max_prob_cat[u_var] * exp_var_mu[n], exp_var_tau[n])
                log_L_var_model += obs_model_var.log_prob(y_n).sum()

            log_L.append(log_L_var_model)

        logging.info(f"Sieved log likelihoods: {log_L}")
        max_elbo_idx = torch.argmax(torch.tensor(log_L))
        return max_elbo_idx

    def compute_elbo(self) -> float:
        if type(self.q) is JointVarDist:
            T_eval, w_T_eval = self.q.t.get_trees_sample()
            self.elbo = self.q.elbo(T_eval, w_T_eval)
        else:
            self.elbo = self.q.elbo()
        return self.elbo

    def step(self):
        self.q.update()
        self.it_counter += 1
        # print info about dist
        logging.debug(str(self.q))
