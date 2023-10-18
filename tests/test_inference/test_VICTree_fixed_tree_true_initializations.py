import os.path
import unittest

import torch

import tests.utils_testing
from inference.victree import VICTree, make_input
from variational_distributions.joint_dists import FixedTreeJointDist
from tests import model_variational_comparisons, utils_testing
from tests.utils_testing import simulate_full_dataset_no_pyro
from utils.config import Config, set_seed
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC


class VICTreeFixedTreeTrueInitializationsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        set_seed(0)

        if not hasattr(self, 'K'):
            self.K = 8
            self.tree = tests.utils_testing.get_tree_K_nodes_random(self.K)
            self.N = 500
            self.M = 1000
            self.A = 7
            self.dir_alpha0 = 3.
            self.nu_0 = 1.
            self.lambda_0 = 100.
            self.alpha0 = 500.
            self.beta0 = 50.
            self.a0 = 10.0
            self.b0 = 1000.0
            y, c, z, pi, mu, tau, eps, eps0, adata = simulate_full_dataset_no_pyro(self.N, self.M, self.A, self.tree,
                                                                                   nu_0=self.nu_0,
                                                                                   lambda_0=self.lambda_0,
                                                                                   alpha0=self.alpha0,
                                                                                   beta0=self.beta0,
                                                                                   a0=self.a0, b0=self.b0,
                                                                                   dir_alpha0=self.dir_alpha0,
                                                                                   return_anndata=True)
            self.y = y
            self.c = c
            self.z = z
            self.pi = pi
            self.mu = mu
            self.tau = tau
            self.eps = eps
            self.eps0 = eps0
            self.adata = adata

    def set_up_q(self, config):
        qc = qC(config)
        qt = qT(config)
        qeps = qEpsilonMulti(config)
        qz = qZ(config)
        qpi = qPi(config)
        qmt = qMuTau(config)
        return qc, qt, qeps, qz, qpi, qmt

    def test_init_true_Z(self):
        """
        If step size small enough, the algorithm shouldn't leave optima of true Z during optimization and ARI should
        be close to 1 (not necessarily = 1 as data generation is stochastic).
        """

        y, c, z, pi, mu, tau, eps, eps0 = (self.y, self.c, self.z, self.pi, self.mu, self.tau, self.eps, self.eps0)

        config = Config(n_nodes=self.K, n_states=self.A, n_cells=self.N, chain_length=self.M, step_size=0.1,
                        diagnostics=False, annealing=1.)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self.id().replace(".", "/"),
                                                                       base_dir='./test_output')

        # qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        # q = FixedTreeJointDist(y, config, qc, qz, qeps, qmt, qpi, self.tree)
        # q.initialize()
        config, q, dh = make_input(data=self.adata, cc_layer=None, config=config,
                                   fix_tree=self.tree, mt_prior_strength=3., eps_prior_strength=0.1,
                                   delta_prior_strength=0.1, step_size=0.3)

        # Initialize Z to true values.
        q.z.initialize('fixed', pi_init=torch.nn.functional.one_hot(z, num_classes=self.K))

        ari_init, best_perm_init, accuracy_best_init = model_variational_comparisons.compare_qZ_and_true_Z(z, q.z)
        self.assertTrue(ari_init > 0.99)

        eta1, eta2 = q.c.update_CAVI(y, q.eps, q.z, q.mt, [self.tree], [1.0])
        q.c.eta1 = eta1
        q.c.eta2 = eta2
        q.c.compute_filtering_probs()

        victree = VICTree(config, q, data_handler=dh, elbo_rtol=1e-4)
        victree.run(n_iter=100)

        # Assert
        torch.set_printoptions(precision=2)
        out = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=c, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps,
                                                                q_c=victree.q.c,
                                                                q_z=victree.q.z, qpi=victree.q.pi,
                                                                q_mt=victree.q.mt, q_eps=victree.q.eps,
                                                                perm=list(range(0, self.K)))
        ari, vscore, perm, acc = (out['ari'], out['v_meas'], out['perm'], out['acc'])

        c_tot_remapped = c[perm]
        z_remapped = torch.tensor([perm[i] for i in z])
        utils_testing.write_inference_test_output(victree, y, c_tot_remapped, z_remapped, self.tree, mu, tau, eps, eps0,
                                                  pi,
                                                  test_dir_path=test_dir_name, file_name_prefix='z_init_')

        self.assertGreater(vscore, 0.90, msg='v-measure less than 0.90')

    def test_init_true_mt_and_C(self):
        y, c, z, pi, mu, tau, eps, eps0 = (self.y, self.c, self.z, self.pi, self.mu, self.tau, self.eps, self.eps0)

        out_dir = "./test_output/" + self._testMethodName
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        config = Config(n_nodes=self.K, n_states=self.A, n_cells=self.N, chain_length=self.M, step_size=0.3,
                        diagnostics=False, out_dir=out_dir, annealing=1.)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self.id().replace(".", "/"))

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)

        q = FixedTreeJointDist(y, config, qc, qz, qeps, qmt, qpi, self.tree)
        # initialize all var dists
        q.initialize()

        # Initialize qMuTau to true values.
        qmt.initialize('fixed', loc=mu, precision_factor=100., shape=self.alpha0, rate=self.alpha0 / tau)
        utils_testing.initialize_qc_to_true_values(c, self.A, qc)
        utils_testing.initialize_qepsilon_to_true_values(eps, self.a0, self.b0, qeps)

        # Make sure qZ is updated first based on good values
        qz.pi = qz.update_CAVI(qmt, qc, qpi, y)

        copy_tree = VICTree(config, q, y, draft=True)
        copy_tree.run(n_iter=50)

        # Assert
        torch.set_printoptions(precision=2)
        out = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=c, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps,
                                                                q_c=copy_tree.q.c,
                                                                q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                                                                q_mt=copy_tree.q.mt, q_eps=copy_tree.q.eps,
                                                                perm=list(range(0, self.K)))
        ari, perm, acc = (out['ari'], out['perm'], out['acc'])

        self.assertGreater(ari, 0.95, msg='ari less than 0.95.')
