import unittest

import torch

from simul import simulate_full_dataset, generate_dataset_var_tree
from utils.config import Config, set_seed
from variational_distributions.var_dists import qC, qEpsilonMulti, qT, qMuTau, qCMultiChrom


class InitTestCase(unittest.TestCase):

    def setUp(self) -> None:
        set_seed(42)

    def test_baum_welch_cluster_init(self):
        config = Config(n_nodes=4, n_states=5, n_cells=100, chain_length=200, wis_sample_size=100, debug=True)
        data = simulate_full_dataset(config)
        # get trees
        fix_qt = qT(config, true_params={
            "tree": data['tree']
        })
        trees_sample, trees_weights = fix_qt.get_trees_sample()

        # get eps
        fix_qeps = qEpsilonMulti(config, true_params={
            "eps": data['eps']
        })

        # random init
        qc_rand = qC(config).initialize(method='random')
        rand_elbo = qc_rand.compute_elbo(trees_sample, trees_weights, fix_qeps)

        # Baum-Welch init
        qc_bw = qC(config).initialize(method='bw-cluster', obs=data['obs'], clusters=data['z'])
        bw_elbo = qc_bw.compute_elbo(trees_sample, trees_weights, fix_qeps)

        # print(bw_elbo, rand_elbo)
        self.assertGreater(bw_elbo, rand_elbo)

        fix_qc = qC(config, true_params={
            "c": data['c']
        })

    @unittest.skip("clonal init not working")
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
        data_elbo = qeps_data_init.compute_elbo([data['tree']], [1.]) +\
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
        qmt_fixed_init = qMuTau(config).initialize(method='fixed', loc=1., precision_factor=2 * config.chain_length,
                                                   shape=config.chain_length, rate=config.chain_length)

        # save init partial elbo and perform one full update using the true complementary distributions
        fixed_init_elbo = qmt_fixed_init.compute_elbo()
        qmt_fixed_init.update(joint_q.c, joint_q.z, joint_q.obs)
        new_elbo = qmt_fixed_init.compute_elbo()
        fix_init_change = - torch.abs(new_elbo - fixed_init_elbo) / new_elbo

        qmt_data_init = qMuTau(config).initialize(method='data', obs=joint_q.obs)
        data_init_elbo = qmt_data_init.compute_elbo()
        qmt_data_init.update(joint_q.c, joint_q.z, joint_q.obs)
        new_elbo = qmt_data_init.compute_elbo()
        data_init_change = - torch.abs(new_elbo - data_init_elbo) / new_elbo

        # larger elbo change after one update means that the distribution was further away from an optimum
        print(f"fixed init change: {fix_init_change}")
        print(f"data init change: {data_init_change}")
        self.assertLess(fix_init_change, data_init_change, "data (mean and var) init distribution "
                                                           "was closer to first update than data init distr")

    def test_true_params_init(self):
        config = Config(n_nodes=5, n_states=7, n_cells=200, chain_length=500, wis_sample_size=20, debug=True)
        joint_q = generate_dataset_var_tree(config)
        # FIXME: computation of elbo in the "fixed" distr setting might be wrong, hence lower elbo
        #   or in general, could disrupt end result
        true_elbo = joint_q.mt.compute_elbo()
        print(joint_q.mt)
        print(true_elbo)

        qmt = qMuTau(config).initialize(method='fixed')
        fix_init_elbo = qmt.compute_elbo()
        print(qmt)
        print(fix_init_elbo)
        qmt = qMuTau(config).initialize(method='data', obs=joint_q.obs)
        data_init_elbo = qmt.compute_elbo()
        print(qmt)
        print(data_init_elbo)

        self.assertTrue(true_elbo > data_init_elbo > fix_init_elbo, msg="elbo of true distribution should be maximum"
                                                                        " and fixed init lowest elbo")


if __name__ == '__main__':
    unittest.main()
