import unittest

import torch

from simul import simulate_full_dataset, generate_dataset_var_tree
from utils.config import Config, set_seed
from variational_distributions.var_dists import qC, qEpsilonMulti, qT, qMuTau, qCMultiChrom


class InitTestCase(unittest.TestCase):

    def setUp(self) -> None:
        set_seed(42)

    # @unittest.skip("clonal init not working")
    def test_eps_init_from_data(self):
        config = Config(n_nodes=5, n_cells=300, chain_length=1000, step_size=.2, debug=True)
        data = simulate_full_dataset(config,
                                     eps_a=10., eps_b=4000., mu0=1., lambda0=3., alpha0=2500., beta0=50.,
                                     cne_length_factor=200, dir_delta=10.)

        qeps_default_init = qEpsilonMulti(config).initialize()
        qeps_data_init = qEpsilonMulti(config).initialize(method='data', obs=data['obs'])

        qc = qCMultiChrom(config)
        qc.initialize(method='clonal', obs=data['obs'])

        default_elbo = qeps_data_init.compute_elbo([data['tree']], [1.]) + \
                       qc.compute_elbo([data['tree']], [1.], qeps_default_init)
        data_elbo = qeps_data_init.compute_elbo([data['tree']], [1.]) + \
                    qc.compute_elbo([data['tree']], [1.], qeps_data_init)

        self.assertGreater(data_elbo, default_elbo)

    def test_mutau_init_from_data(self):
        """
        Test proves that mutau should be initialized to values that are in the scale of
        the data size. Even though data init uses data mean and variance, the parameters
        of mu-tau dist will be updated to reach values that are in the data size scale
        (see mu-tau updates formulas).
        """
        config = Config(n_nodes=4, n_states=5, n_cells=100, chain_length=200, wis_sample_size=100, debug=True)
        joint_q = generate_dataset_var_tree(config)

        # fix to data size dependent factors
        qmt_size_init = qMuTau(config, nu_prior=1., lambda_prior=config.chain_length,
                               alpha_prior=config.chain_length, beta_prior=config.chain_length / 10.)
        qmt_size_init.initialize(method='data-size', obs=joint_q.obs)

        # save init partial elbo and perform one full update using the true complementary distributions
        fixed_init_elbo = qmt_size_init.compute_elbo()
        qmt_size_init.update(joint_q.c, joint_q.z, joint_q.obs)
        new_elbo = qmt_size_init.compute_elbo()
        fix_init_change = - torch.abs(torch.tensor(new_elbo - fixed_init_elbo)) / new_elbo

        qmt_data_init = qMuTau(config, nu_prior=1., lambda_prior=config.chain_length,
                               alpha_prior=config.chain_length, beta_prior=config.chain_length / 10.)
        qmt_data_init.initialize(method='data', obs=joint_q.obs)
        data_init_elbo = qmt_data_init.compute_elbo()
        qmt_data_init.update(joint_q.c, joint_q.z, joint_q.obs)
        new_elbo = qmt_data_init.compute_elbo()
        data_init_change = - torch.abs(torch.tensor(new_elbo - data_init_elbo)) / new_elbo

        # larger elbo change after one update means that the distribution was further away from an optimum
        print(f"fixed init change: {fix_init_change}")
        print(f"data init change: {data_init_change}")
        self.assertLess(fix_init_change, data_init_change, "data (mean and var) init distribution "
                                                           "was closer to first update than data size init distr")


if __name__ == '__main__':
    unittest.main()
