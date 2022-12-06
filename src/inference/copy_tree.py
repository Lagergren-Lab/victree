from numpy import infty
import torch

from utils.config import Config
from model.generative_model import GenerativeModel
from variational_distributions.variational_distribution import VariationalDistribution
from variational_distributions.var_dists import qT, qEpsilon, qMuTau, qPi, qZ, qC


class JointVarDist(VariationalDistribution):
    def __init__(self, config: Config,
                 qc, qz, qt, qeps, qmt, qpi, obs: torch.Tensor):
        super().__init__(config)
        self.c: qC = qc
        self.z: qZ = qz
        self.t: qT = qt
        self.eps: qEpsilon = qeps
        self.mt: qMuTau = qmt
        self.pi: qPi = qpi
        self.obs = obs

    def update(self, p: GenerativeModel):
        # T, C, eps, z, mt, pi
        trees, weights = self.t.get_trees_sample()
        self.t.update(trees, self.c.couple_filtering_probs, self.c, self.eps)
        self.c.update(self.obs, self.t, self.eps, self.z, self.mt)
        self.eps.update(trees, weights, self.c.couple_filtering_probs)
        self.pi.update(self.z, p.delta_pi)
        self.z.update(self.mt, self.c, self.pi, self.obs)
        self.mt.update()
        # TODO: continue

        return super().update()

    def initialize(self):
        for q in [self.t, self.c, self.eps, self.pi, self.z, self.mt]:
            q.initialize()
        return super().initialize()

    def elbo(self) -> float:
        return self.c.elbo() +\
                self.z.elbo() +\
                self.mt.elbo() +\
                self.pi.elbo() +\
                self.eps.elbo() +\
                self.t.elbo()

class CopyTree():

    def __init__(self, config: Config,
                 p: GenerativeModel,
                 q: JointVarDist,
                 obs: torch.Tensor):

        self.config = config
        self.p = p
        self.q = q
        self.obs = obs

        # counts the number of steps performed
        self.it_counter = 0    
        self.elbo = -infty

    def run(self, n_iter):

        # counts the number of irrelevant updates
        close_runs = 0

        self.init_variational_variables()

        for _ in range(n_iter):
            # do the updates
            self.step()

            old_elbo = self.elbo
            self.compute_elbo()
            if abs(old_elbo - self.elbo) < self.config.elbo_tol:
                close_runs += 1
                if close_runs > self.config.max_close_runs:
                    break
            elif self.elbo < old_elbo:
                # elbo should only increase
                raise ValueError("Elbo is decreasing")
            elif self.elbo > 0:
                # elbo must be negative
                raise ValueError("Elbo is non-negative")
            else:
                close_runs = 0

    def compute_elbo(self) -> float:
        # TODO: elbo could also be a custom object, containing the main elbo parts separately
        #   so we can monitor all components of the elbo (variational and model part)

        # TODO: maybe parallelize elbos computations
        elbo_q = self.q.elbo()
        # FIXME: entropy not implemented yet
        elbo_p = 0. # entropy

        self.elbo = elbo_q + elbo_p
        return self.elbo

    def step(self):
        self.update_T()
        self.update_C(self.obs)
        self.q.c.calculate_filtering_probs()
        self.update_z()
        self.update_mutau()
        self.update_epsilon()
        self.update_gamma()

        self.it_counter += 1

    def init_variational_variables(self):
        # random initialization of variational parameters

        pass

    def update_T(self):
        pass

    def update_C(self, obs):
        self.q.c.update(obs, self.q.t, self.q.eps, self.q.z, self.q.mt)

    def update_z(self):
        self.q.z.update(self.q.mt, self.q.c, self.q.pi, self.obs)

    def update_mutau(self):
        self.q.mt.update()

    def update_epsilon(self):
        trees, weights = self.q.t.get_trees_sample()
        self.q.eps.update(trees, weights, self.q.c.couple_filtering_probs)

    def update_gamma(self):
        pass


