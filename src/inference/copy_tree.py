from typing import Union

import networkx as nx
from numpy import infty
import torch

from utils.config import Config
from model.generative_model import GenerativeModel
from variational_distributions.variational_distribution import VariationalDistribution
from variational_distributions.var_dists import qEpsilonMulti, qT, qEpsilon, qMuTau, qPi, qZ, qC


class JointVarDist(VariationalDistribution):
    def __init__(self, config: Config,
                 qc, qz, qt, qeps, qmt, qpi, obs: torch.Tensor):
        super().__init__(config)
        self.c: qC = qc
        self.z: qZ = qz
        self.t: qT = qt
        self.eps: Union[qEpsilon, qEpsilonMulti] = qeps
        self.mt: qMuTau = qmt
        self.pi: qPi = qpi
        self.obs = obs

    def update(self, p: GenerativeModel):
        # T, C, eps, z, mt, pi
        trees, weights = self.t.get_trees_sample()
        self.t.update(trees, self.c, self.eps)
        self.c.update(self.obs, self.eps, self.z, self.mt, trees, weights)
        self.eps.update(trees, weights, self.c.couple_filtering_probs)
        self.pi.update(self.z)
        self.z.update(self.mt, self.c, self.pi, self.obs)
        self.mt.update(self.c, self.z, self.obs, torch.sum(self.obs ** 2))

        return super().update()

    def initialize(self):
        for q in [self.t, self.c, self.eps, self.pi, self.z, self.mt]:
            q.initialize()
        return super().initialize()

    def elbo(self, T_eval, w_T_eval) -> float:
        return self.c.elbo(T_eval, w_T_eval, self.eps) + \
               self.z.elbo(self.pi) + \
               self.mt.elbo() + \
               self.pi.elbo() + \
               self.eps.elbo() + \
               self.t.elbo()


class VarDistFixedTree(VariationalDistribution):
    def __init__(self, config: Config,
                 qc, qz, qeps, qmt, qpi, T: nx.DiGraph, obs: torch.Tensor):
        super().__init__(config)
        self.c: qC = qc
        self.z: qZ = qz
        self.eps: Union[qEpsilon, qEpsilonMulti] = qeps
        self.mt: qMuTau = qmt
        self.pi: qPi = qpi
        self.obs = obs
        self.T = T
        self.w_T = [1.0]

    def update(self, p: GenerativeModel):
        # T, C, eps, z, mt, pi
        self.c.update(self.obs, self.eps, self.z, self.mt, [self.T], self.w_T)
        self.eps.update([self.T], self.w_T, self.c.couple_filtering_probs)
        self.pi.update(self.z)
        self.z.update(self.mt, self.c, self.pi, self.obs)
        self.mt.update(self.c, self.z, self.obs, torch.sum(self.obs ** 2))

        return super().update()

    def initialize(self):
        for q in [self.c, self.eps, self.pi, self.z, self.mt]:
            q.initialize()
        return super().initialize()

    def elbo(self) -> float:
        q_C_elbo = self.c.elbo([self.T], self.w_T, self.eps)
        q_Z_elbo = self.z.elbo(self.pi)
        q_MuTau_elbo = self.mt.elbo()
        q_pi_elbo = self.pi.elbo()
        q_eps_elbo = self.eps.elbo()
        return q_Z_elbo + q_MuTau_elbo + q_pi_elbo + q_eps_elbo + q_C_elbo


class CopyTree():

    def __init__(self, config: Config,
                 p: GenerativeModel,
                 q: Union[JointVarDist, VarDistFixedTree],
                 obs: torch.Tensor):

        self.config = config
        self.p = p
        self.q = q
        self.obs = obs

        # non-mutable
        self.sum_over_m_y_squared = torch.sum(self.obs ** 2)

        # counts the number of steps performed
        self.it_counter = 0
        self.elbo = -infty

    def run(self, n_iter):

        # counts the number of irrelevant updates
        close_runs = 0

        self.init_variational_variables()
        self.compute_elbo()
        print(f"ELBO after init: {self.elbo}")

        for _ in range(n_iter):
            # do the updates
            self.step()

            old_elbo = self.elbo
            self.compute_elbo()
            print(f"ELBO: {self.elbo}")
            if abs(old_elbo - self.elbo) < self.config.elbo_tol:
                close_runs += 1
                if close_runs > self.config.max_close_runs:
                    break
            elif self.elbo < old_elbo:
                # elbo should only increase
                # raise ValueError("Elbo is decreasing")
                print("Elbo is decreasing")
            elif self.elbo > 0:
                # elbo must be negative
                # raise ValueError("Elbo is non-negative")
                print("Warning: Elbo is non-negative")
            else:
                close_runs = 0

    def compute_elbo(self) -> float:
        if type(self.q) is JointVarDist:
            T_eval, w_T_eval = self.q.t.get_trees_sample()
            self.elbo = self.q.elbo(T_eval, w_T_eval)
        else:
            self.elbo = self.q.elbo()
        return self.elbo

    def step(self):
        self.q.update(self.p)
        self.it_counter += 1

    def init_variational_variables(self):
        self.q.initialize()
