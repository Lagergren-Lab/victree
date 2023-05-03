import copy
import logging
import math
import random
from typing import Union, List

import numpy as np
import torch
import torch.distributions as dist
from tqdm import tqdm

from utils.config import Config
from variational_distributions.joint_dists import VarTreeJointDist, FixedTreeJointDist


class CopyTree:

    def __init__(self, config: Config,
                 q: Union[VarTreeJointDist, FixedTreeJointDist],
                 obs: torch.Tensor):

        self.config = config
        self.q = q
        self.obs = obs

        # counts the number of steps performed
        self.it_counter = 0
        self._elbo: float = -np.infty
        self.sieve_models: List[Union[VarTreeJointDist, FixedTreeJointDist]] = []

    def __str__(self):
        return f"k{self.config.n_nodes}"\
               f"a{self.config.n_states}"\
               f"n{self.config.n_cells}"\
               f"m{self.config.chain_length}"

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

            curr_elbo = curr_model.compute_elbo()
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
            elbos[i] = model.compute_elbo()

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

                curr_elbo = m.compute_elbo()
                logging.info(f"[S{i}] elbo: {curr_elbo} at final iter ({start_iter + step_iters})")
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
            elbos.append(self.sieve_models[i].compute_elbo())

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
        if type(self.q) is VarTreeJointDist:
            T_eval, w_T_eval = self.q.t.get_trees_sample()
            self.elbo = self.q.compute_elbo(T_eval, w_T_eval)
        else:
            self.elbo = self.q.compute_elbo()
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
