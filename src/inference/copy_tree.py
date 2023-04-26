import copy
import logging
import math
import random
from typing import Union, List

import networkx as nx
import numpy as np
import torch
import torch.distributions as dist
from tqdm import tqdm

from utils import math_utils
from utils.config import Config, set_seed
from utils.tree_utils import tree_to_newick
from variational_distributions.observational_variational_distribution import qPsi
from variational_distributions.variational_distribution import VariationalDistribution
from variational_distributions.var_dists import qEpsilonMulti, qT, qEpsilon, qMuTau, qPi, qZ, qC, \
    qMuAndTauCellIndependent, qPhi


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

        self.diagnostics_dict = {} if config.diagnostics else None

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

        L = self.config.wis_sample_size
        self.diagnostics_dict["wG"] = torch.zeros((n_iter, K, K))
        self.diagnostics_dict["wT"] = torch.zeros((n_iter, L))
        self.diagnostics_dict["gT"] = torch.zeros((n_iter, L))
        self.diagnostics_dict["T"] = []

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

        self.diagnostics_dict["wG"][iter, ...] = torch.tensor(nx.to_numpy_array(self.t.weighted_graph))
        self.diagnostics_dict["wT"][iter] = self.t.w_T
        self.diagnostics_dict["gT"][iter] = self.t.g_T
        self.diagnostics_dict["T"].append([tree_to_newick(t) for t in self.t.get_trees_sample()[0]])


class VarDistFixedTree(VariationalDistribution):
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

        self.diagnostics_dict = {} if config.diagnostics else None

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


class CopyTree:

    def __init__(self, config: Config,
                 q: Union[JointVarDist, VarDistFixedTree],
                 obs: torch.Tensor):

        self.config = config
        self.q = q
        self.obs = obs

        # counts the number of steps performed
        self.it_counter = 0
        self._elbo: float = -np.infty
        self.sieve_models: List[Union[JointVarDist, VarDistFixedTree]] = []

    @property
    def elbo(self):
        return self._elbo

    @elbo.setter
    def elbo(self, e):
        if isinstance(e, torch.Tensor):
            self._elbo = e.item()
        else:
            self._elbo = e

    def run(self, n_iter):

        # counts the number of irrelevant updates
        close_runs = 0

        if self.q.diagnostics_dict is not None:
            self.q.init_diagnostics(self.config.n_sieving_iter + n_iter + 1)  # n_iter + 1 for initialization values
            self.q.update_diagnostics(0)

        if self.config.sieving_size > 1:
            logging.info(f"Sieving {self.config.sieving_size} runs with "
                         f"{self.config.n_sieving_iter} iterations")
            logging.info(f"ELBO before sieving: {self.compute_elbo():.2f}")

            # run inference on separate initialization of copytree and select the best one
            # TODO: add initialization parameters
            # self.sieve(ktop=3)
            self.halve_sieve()
            logging.info(f"ELBO after sieving: {self.compute_elbo():.2f}")

        else:
            logging.info(f"ELBO after init: {self.compute_elbo():.2f}")

        logging.info("Start VI updates")
        pbar = tqdm(range(1, n_iter + 1))
        for it in pbar:
            # do the updates
            self.step()
            if self.config.annealing != 1.0:
                self.set_temperature(it, n_iter)

            old_elbo = self.elbo
            self.compute_elbo()

            pbar.set_postfix({'elbo': self.elbo})
            if self.q.diagnostics_dict is not None:
                self.q.update_diagnostics(it + self.config.n_sieving_iter)

            if abs((old_elbo - self.elbo) / self.elbo) < self.config.elbo_rtol:
                close_runs += 1
                if close_runs > self.config.max_close_runs:
                    logging.debug(f"*** Run ended after {it}/{n_iter} iterations due to plateau ***")
                    break
            elif self.elbo < old_elbo:
                # elbo should only increase
                logging.warning("Elbo is decreasing")
            else:
                close_runs = 0

        logging.info(f"ELBO final: {self.elbo:.2f}")

    def topk_sieve(self, ktop: int = 1, **kwargs):
        """
        Creates self.config.sieving_size number of copies of self.q, initializes each q differently,
        performs n_sieve_iter updates of q, calculates the ELBO of each copy and sets self.q to one of
        the top-k best, sampled using the ELBO as weight.
        :param n_sieve_iter: number of updates before sieving selection
        :return:

        Parameters
        ----------
        ktop specifies how many instances of q to keep, among which to sample
        **kwargs key,value pairs for init params
        """
        # TODO: parallelize this for loop (make sure that randomness is properly split)
        logging.info("topk-sieving started")
        top_k_models = [None] * ktop
        top_k_elbos = - np.infty * np.ones(ktop)
        for i in range(self.config.sieving_size):
            curr_model = copy.deepcopy(self.q)
            curr_model.initialize(**kwargs)

            logging.info(f"[S{i}] started")
            for j in tqdm(range(1, self.config.n_sieving_iter + 1)):
                curr_model.update()
                if self.config.diagnostics:
                    curr_model.update_diagnostics(j)

            curr_elbo = curr_model.elbo()
            logging.info(f"[S{i}] elbo: {curr_elbo} at final iter ({self.config.n_sieving_iter})")
            if np.any(top_k_elbos < curr_elbo):
                logging.info("new top model!")
                sel = np.argmin(top_k_elbos)  # take lowest elbo index
                top_k_elbos[sel] = curr_elbo
                top_k_models[sel] = curr_model

        # sample one among top k based on elbo
        sel_idx = random.choices(range(len(top_k_models)), weights=top_k_elbos / top_k_elbos.sum(), k=1)[0]
        logging.info(f"[siev] selected {sel_idx} among top-{ktop}")
        self.q = top_k_models[sel_idx]

    def halve_sieve(self, **kwargs) -> None:
        """
        Runs config.sieving_size instances from different initializations and progressively
        halves the instances, keeping the best ones, until only one remains.
        The number of sieving iterations (config.n_sieving_iters) is divided on each step
        e.g. sieving_size = 8, n_sievin_iters = 11
            8 runs for 3 iters -> 4 runs for 3 iters -> 2 runs for 5 iters -> 1 best out
        """
        logging.info("halve-sieving started")
        # outer piece of recursive halve_sieve
        models = []
        elbos = np.zeros(self.config.sieving_size)
        for i in range(self.config.sieving_size):
            model = copy.deepcopy(self.q)
            model.initialize(**kwargs)
            models.append(model)
            elbos[i] = model.elbo()

        # reduce sieving iterations per step so that total sieving iterations
        # match the number specified in config
        # e.g. 10 runs, 100 iterations
        #   -> gets halved log2(10) = 3 times
        #   -> each time runs for 100 // 3 = 33 iterations
        # start recursion
        self.q = self.halve_sieve_r(models, elbos)

    def halve_sieve_r(self, models, elbos, start_iter: int = 1):
        if len(models) > 1:
            # k: number of further selections
            k = len(models) // 2
            num_sieve_steps = math.floor(math.log2(self.config.sieving_size))
            step_iters = self.config.n_sieving_iter // num_sieve_steps
            # make the last sieve step lasts until reaching the total sieving num iterations exactly
            # if we are selecting the top 1, then make the last instances run for the remaining num of iterations
            if k < 2 and (step_iters * num_sieve_steps) < self.config.n_sieving_iter:
                step_iters += self.config.n_sieving_iter - step_iters * num_sieve_steps  # add remaining iterations

            logging.info(f"[siev]: halving models (k:  {len(models)} -> {k})")
            # selected models
            sel_models = [None] * k
            sel_elbos = - np.infty * np.ones(k)
            for i, m in enumerate(models):
                logging.info(f"[S{i}] started")
                for j in tqdm(range(start_iter, start_iter + step_iters)):
                    m.update()
                    if self.config.diagnostics:
                        m.update_diagnostics(j)

                curr_elbo = m.elbo()
                logging.info(f"[S{i}] elbo: {curr_elbo} at final iter ({self.config.n_sieving_iter})")
                if np.any(sel_elbos < curr_elbo):
                    logging.info("new top model!")
                    sel = np.argmin(sel_elbos)  # take lowest elbo index
                    sel_elbos[sel] = curr_elbo
                    sel_models[sel] = m

            return self.halve_sieve_r(sel_models, sel_elbos, start_iter=start_iter+step_iters)
        else:
            final_model_idx = np.argmax(elbos)
            logging.info(f"[siev] selected model with elbo: {elbos[final_model_idx]}")
            return models[final_model_idx]

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
        # print info about dist every 10 it
        if self.it_counter % 10 == 0:
            logging.debug(str(self.q))

    def set_temperature(self, it, n_iter):
        # linear scheme: from annealing to 1 with equal steps between iterations
        self.q.z.temp = self.config.annealing - (it - 1)/(n_iter - 1) * (self.config.annealing - 1.)
        self.q.mt.temp = self.config.annealing - (it - 1)/(n_iter - 1) * (self.config.annealing - 1.)
