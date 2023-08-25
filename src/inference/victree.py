import copy
import json
import logging
import math
import os
import random
from typing import Union, List

import anndata
import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
from tqdm import tqdm

from inference.split_and_merge_operations import SplitAndMergeOperations
from utils.config import Config
from utils.data_handling import write_output, DataHandler
from variational_distributions.joint_dists import VarTreeJointDist, FixedTreeJointDist
from variational_distributions.var_dists import qCMultiChrom


class VICTree:

    def __init__(self, config: Config,
                 q: Union[VarTreeJointDist, FixedTreeJointDist],
                 obs: torch.Tensor, data_handler: DataHandler | None = None,
                 draft=False):
        """
        Inference class, to set up, run and store results of inference.
        Parameters
        ----------
        config: Config configuration object
        q: JointDist (Var- or Fixed- tree) variational joint distribution
        obs: torch.Tensor of shape (n_sites, n_cells) observation matrix
        """

        self.config = config
        self.q = q
        self.obs = obs
        self._draft = draft  # if true, does not save output on file

        # counts the number of steps performed
        self.it_counter = 0
        self._elbo: float = -np.infty
        self.sieve_models: List[Union[VarTreeJointDist, FixedTreeJointDist]] = []

        if data_handler is None:
            adata = anndata.AnnData(X=obs.T.numpy(),
                                    layers={'copy': obs.T.numpy()},
                                    var=pd.DataFrame({
                                        'chr': [1] * self.config.chain_length,
                                        'start': [i for i in range(self.config.chain_length)],
                                        'end': [i + 1 for i in range(self.config.chain_length)]
                                    }))
            data_handler = DataHandler(adata=adata)
        self._data_handler: DataHandler = data_handler
        if self.config.split:
            self.split_operation = SplitAndMergeOperations()

    def __str__(self):
        return f"k{self.config.n_nodes}" \
               f"a{self.config.n_states}" \
               f"n{self.config.n_cells}" \
               f"m{self.config.chain_length}"

    @property
    def elbo(self):
        return self._elbo

    @elbo.setter
    def elbo(self, e):
        if isinstance(e, torch.Tensor):
            # to keep _elbo as float primitive type
            self._elbo = e.item()
        else:
            self._elbo = e

    @property
    def cache_size(self):
        # returns the number of iterations that are currently saved in the params_history variables
        assert 'elbo' in self.q.params_history
        return len(self.q.params_history['elbo'])

    @property
    def data_handler(self):
        return self._data_handler

    def run(self, n_iter=-1, args=None):
        """
        Set-up diagnostics, run sieving and perform VI steps, checking elbo for early-stopping.
        Parameters
        ----------
        n_iter: int, number of iterations (after sieving iterations, if any)
        args: dict, parsed execution arguments
        """

        # TODO: clean if-case and use just config param
        if n_iter == -1:
            n_iter = self.config.n_run_iter
        z_init = "random"
        if args is not None:
            z_init = args.z_init

        # ---
        # Diagnostic object setup
        # ---
        checkpoint_path = os.path.join(self.config.out_dir, "victree.diagnostics.h5")
        if self.config.diagnostics:
            # check if checkpoint already exists, then print warning
            if os.path.exists(checkpoint_path):
                logging.warning("diagnostic file already exists, will be overwritten")
                os.remove(checkpoint_path)

        # counts the number of converged updates
        close_runs = 0

        # ---
        # Sieving: run few iterations with multiple initializations and continue with the most promising
        # ---
        if self.config.sieving_size > 1:
            logging.info(f"Sieving {self.config.sieving_size} runs with "
                         f"{self.config.n_sieving_iter} iterations")
            logging.info(f"ELBO before sieving: {self.compute_elbo():.2f}")

            # run inference on separate initialization of copytree and select the best one
            # TODO: add initialization parameters
            # self.sieve(ktop=3)
            self.halve_sieve(z_init=z_init, obs=self.obs)
            logging.info(f"ELBO after sieving: {self.compute_elbo():.2f}")

        else:
            logging.info(f"ELBO after init: {self.compute_elbo():.2f}")

        # ---
        # Main run
        # ---
        logging.info("Start VI updates")
        pbar = tqdm(range(1, n_iter + 1))
        for it in pbar:
            # KEY inference algorithm iteration step
            if self.config.split:
                self.split()
            self.step()

            # update all the other meta-parameters
            if self.config.annealing != 1.0:
                # annealing for different update step-size along iterations
                # FIXME: currently this temperature does not change any setting
                self.set_temperature(it, n_iter)

            old_elbo = self.elbo
            self.compute_elbo()

            # progress bar showing elbo
            pbar.set_postfix({'elbo': self.elbo})

            # early-stopping
            if np.abs((self.elbo - old_elbo) / self.elbo) < self.config.elbo_rtol * self.config.step_size:
                close_runs += 1
                if close_runs > self.config.max_close_runs:
                    logging.debug(f"run converged after {it}/{n_iter} iterations")
                    break
            elif self.elbo < old_elbo:
                # elbo should only increase
                close_runs += 1
                logging.warning("Elbo is decreasing")
            else:
                close_runs = 0

            if it % self.config.save_progress_every_niter == 0 and self.config.diagnostics:
                self.write()

        logging.info(f"ELBO final: {self.elbo:.2f}")
        # write last chunks of output to diagnostics
        if self.config.diagnostics:
            self.write_checkpoint_h5(path=checkpoint_path)

    def topk_sieve(self, ktop: int = 1, **kwargs):
        """
        Creates self.config.sieving_size number of copies of self.q, initializes each q differently,
        performs n_sieve_iter updates of q, calculates the ELBO of each copy and sets self.q to one of
        the top-k best, sampled using the ELBO as weight.

        Parameters
        ----------
        ktop: int, specifies how many instances of q to keep, among which to sample
        **kwargs key,value pairs for initialization parameters
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
        # recursive function
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

                curr_elbo = m.compute_elbo()
                logging.info(f"[S{i}] elbo: {curr_elbo} at final iter ({start_iter + step_iters})")
                if np.any(sel_elbos < curr_elbo):
                    logging.info("new top model!")
                    sel = np.argmin(sel_elbos)  # take lowest elbo index
                    sel_elbos[sel] = curr_elbo
                    sel_models[sel] = m

            return self.halve_sieve_r(sel_models, sel_elbos, start_iter=start_iter + step_iters)
        else:
            final_model_idx = np.argmax(elbos)
            logging.info(f"[siev] selected model with elbo: {elbos[final_model_idx]}")
            return models[final_model_idx]

    # deprecated
    def sieving_selection_ELBO(self):
        elbos = []
        for i in range(self.config.sieving_size):
            elbos.append(self.sieve_models[i].compute_elbo())

        logging.info(f"Sieved elbos: {elbos}")
        max_elbo_idx = torch.argmax(torch.tensor(elbos))
        return max_elbo_idx

    # deprecated
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
        """
        Compute ELBO and saves it to attribute
        Returns
        -------
        elbo, float
        """
        if type(self.q) is VarTreeJointDist:
            # evaluate elbo with tree samples from the tree variational distribution
            T_eval, w_T_eval = self.q.t.get_trees_sample()
            self.elbo = self.q.compute_elbo(T_eval, w_T_eval)
        else:
            # or in a fixed tree
            self.elbo = self.q.compute_elbo()
        return self.elbo

    def step(self, diagnostics_path: str = None):
        """
        Wrapper function for variational updates. Handles checkpoint saving.
        """
        if self.config.SVI:
            self.q.SVI_update(self.it_counter)
        else:
            self.q.update(self.it_counter)
        self.it_counter += 1
        # print info about dist every 10 it
        if self.it_counter % 10 == 0:
            logging.debug(f"-------- it: {self.it_counter} --------")
            logging.debug(str(self.q))
        # save checkpoint every 20 iterations
        if self.config.diagnostics and self.it_counter % self.config.save_progress_every_niter == 0:
            self.write_checkpoint_h5(path=diagnostics_path)
        self.config.curr_it += 1

    def set_temperature(self, it, n_iter):
        # linear scheme: from annealing to 1 with equal steps between iterations
        self.q.z.temp = self.config.annealing - (it - 1) / (n_iter - 1) * (self.config.annealing - 1.)
        self.q.mt.temp = self.config.annealing - (it - 1) / (n_iter - 1) * (self.config.annealing - 1.)

    def write_model(self, path: str):
        # save victree distributions parameters
        with h5py.File(path, 'w') as f:
            for q in self.q.get_units() + [self.q]:
                qlay = f.create_group(q.__class__.__name__)
                params = q.get_params_as_dict()
                prior_params = q.get_prior_params_as_dict()
                prior_params = {} if prior_params is None else prior_params
                if isinstance(q, qCMultiChrom):
                    # get params gives dict[str, list[np.ndarray]] for each unit qC
                    for i, qc in enumerate(q.qC_list):
                        qclay = qlay.create_group(qc.chromosome_name)
                        for k in params:
                            qclay.create_dataset(k, data=params[k][i])
                # regular distribution
                else:
                    for k in params:
                        qlay.create_dataset(k, data=params[k])
                    for k in prior_params:
                        if k:
                            qlay.create_dataset(k, data=prior_params[k])
        logging.debug(f"model saved in {path}")

    def write(self):
        if not self._draft:
            # write output in anndata
            out_anndata_path = os.path.join(self.config.out_dir, 'victree.out.h5ad')
            write_output(self, out_anndata_path, anndata=True)

            # write output model
            out_model_path = os.path.join(self.config.out_dir, 'victree.model.h5')
            self.write_model(out_model_path)

            # write configuration to json
            out_json_path = os.path.join(self.config.out_dir, 'victree.config.json')
            with open(out_json_path, 'w') as jsonf:
                json.dump(self.config.to_dict(), jsonf)
            logging.debug(f"config saved in {out_json_path}")
        else:
            logging.debug(f"output saving skipped due to `draft` flag set to True")

    def write_checkpoint_h5(self, path=None):
        if self.cache_size > 0:
            if path is None:
                path = os.path.join(self.config.out_dir, "victree.diagnostics.h5")

            # append mode, so that if the file already exist, then the data is appended
            with h5py.File(path, 'a') as f:
                # for each of the individual q dist + the joint dist itself (e.g. to monitor joint_q.elbo)
                if len(f.keys()) == 0:
                    # init h5 file
                    for q in self.q.get_units() + [self.q]:
                        qlay = f.create_group(q.__class__.__name__)
                        for k in q.params_history.keys():
                            stacked_arr = np.stack(q.params_history[k], axis=0)
                            # init dset with unlimited number of iteration and fix other dims
                            ds = qlay.create_dataset(k, data=stacked_arr,
                                                     maxshape=(
                                                         self.config.n_sieving_iter + self.config.n_run_iter + 1,
                                                         *stacked_arr.shape[1:]), chunks=True)
                else:
                    # resize and append
                    for q in self.q.get_units() + [self.q]:
                        qlay = f[q.__class__.__name__]
                        for k in q.params_history.keys():
                            stacked_arr = np.stack(q.params_history[k], axis=0)
                            ds = qlay[k]
                            ds.resize(ds.shape[0] + stacked_arr.shape[0], axis=0)
                            ds[-stacked_arr.shape[0]:] = stacked_arr

                # wipe cache
                for q in self.q.get_units() + [self.q]:
                    q.reset_params_history()

            logging.debug(f"diagnostics saved in {path}")

    def split(self):
        split = self.split_operation.split(self.obs, self.q.c, self.q.z, self.q.mt, self.q.pi)
        if split:
            self.q.mt.update(self.q.c, self.q.z, self.q.obs)

