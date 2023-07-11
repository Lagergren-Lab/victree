import logging
import os.path
import unittest

import torch

import simul
import tests.utils_testing
from inference.copy_tree import CopyTree
from variational_distributions.joint_dists import QuadrupletJointDist
from tests import model_variational_comparisons
from tests.utils_testing import simul_data_pyro_full_model, simulate_full_dataset_no_pyro
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent


class VICtreeQuadrupletTestCase(unittest.TestCase):

    def set_up_q(self, config):
        qc = qC(config)
        qeps = qEpsilonMulti(config)
        z_assgn = torch.tensor([2, 3])
        qz = qZ(config, true_params={'z': z_assgn})
        qmt = qMuTau(config, nu_prior=1., lambda_prior=1., alpha_prior=100., beta_prior=5.)
        return qc, qeps, qz, qmt

    def test_quadruplet(self):
        torch.manual_seed(0)
        tree = tests.utils_testing.get_quadtruplet_tree()
        n_cells = 2
        n_sites = 200
        n_copy_states = 7
        nu_0 = torch.tensor(1.)
        lambda_0 = torch.tensor(10.)
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        a0 = torch.tensor(5.0)
        b0 = torch.tensor(200.0)
        out = simul.simulate_quadruplet_data(n_sites, n_copy_states, tree, a0, b0, eps_0=0.1)
        y, c, mu, tau, eps, eps0 = out['obs'], out['c'], out['mu'], out['tau'], out['eps'], out['eps0']
        print(f"C: {c}")

        out_dir = "./test_output/" + self._testMethodName
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        config = Config(n_nodes=4, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.1,
                        diagnostics=False, out_dir=out_dir, annealing=1.)
        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)
        qc, qeps, qz, qmt = self.set_up_q(config)
        q = QuadrupletJointDist(config, qc, qz, qeps, qmt, tree, y)
        qc.initialize()
        qeps.initialize()
        qmt.initialize(loc=1., precision_factor=1., shape=100., rate=10.)
        copy_tree = CopyTree(config, q, y)

        copy_tree.run(n_iter=230)
        copy_tree.step()
        #print(q.c)

        # Assert
        print(q.c.single_filtering_probs.argmax(dim=-1))
        print(c)
        model_variational_comparisons.compare_qC_and_true_C(true_C=c, q_c=copy_tree.q.c)
        model_variational_comparisons.compare_qMuTau_with_true_mu_and_tau(true_mu=mu, true_tau=tau, q_mt=copy_tree.q.mt)
        #model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=c, true_Z=z, true_pi=pi, true_mu=mu,
        #                                                  true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
        #                                                  q_z=copy_tree.q.z, qpi=copy_tree.q.z, q_mt=copy_tree.q.mt)