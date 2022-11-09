from typing import Dict
from numpy import infty
from utils.config import Config
from variational_distributions.q_T import q_T
from variational_distributions.q_Z import qZ
from variational_distributions.q_epsilon import qEpsilon
from variational_distributions.variational_distribution import VariationalDistribution
from variational_distributions.variational_hmm import CopyNumberHmm
from variational_distributions.variational_normal import qMuTau


class CopyTree():

    def __init__(self, p,
            config: Config, 
            qc: CopyNumberHmm,
            qz: qZ,
            qt: q_T,
            qeps: qEpsilon,
            qmt: qMuTau
            ):
        self.config = config
        self.p = p
        self.q = JointVarDist(config, qc, qz, qt, qeps, qmt)

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

            new_elbo = self.compute_elbo()
            if abs(new_elbo - self.elbo) < self.config.elbo_tol:
                close_runs += 1
                if close_runs > self.config.max_close_runs:
                    break
            elif new_elbo < self.elbo:
                # elbo should only increase
                raise ValueError("Elbo is decreasing")
            elif new_elbo > 0:
                # elbo must be negative
                raise ValueError("Elbo is non-negative")
            else:
                close_runs = 0



    def compute_elbo(self) -> float:
        # TODO: elbo could also be a custom object, containing the main elbo parts separately
        #   so we can monitor all components of the elbo (variational and model part)

        # ...quite costly operation...
        return -1000.

    def step(self):
        self.update_T()
        self.update_C()
        self.update_z()
        self.update_mu()
        self.update_sigma()
        self.update_epsilon()
        self.update_gamma()

        self.it_counter += 1

    def init_variational_variables(self):
        # random initialization of variational parameters

        pass

    def update_T(self):
        pass

    def update_C(self, obs):
        self.q.c.update(obs,
                self.q.t, self.q.eps, self.q.z, self.q.mt)

    def update_z(self):
        self.q.z.update()

    def update_mutau(self):
        self.q.mt.update()

    def update_epsilon(self):
        self.q.eps.update(trees, w_T, q_C_pairwise_marginals)

    def update_gamma(self):
        pass


class JointVarDist(VariationalDistribution):
    def __init__(self, config: Config,
            qc, qz, qt, qeps, qmt, obs = None):
        super().__init__(config)
        self.c: CopyNumberHmm = qc
        self.z: qZ = qz
        self.t: q_T = qt
        self.eps: qEpsilon = qeps
        self.mt: qMuTau = qmt
        self.obs = obs

    def update(self):
        # T, C, eps, z, mt, pi
        self.t.update()
        self.c.update()
        self.eps.update()
        self.z.update()
        self.mt.update()
        # TODO: continue

        return super().update()

    def initialize(self):
        return super().initialize()
