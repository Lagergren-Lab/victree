import unittest

from simul import simulate_full_dataset, generate_dataset_var_tree
from utils.config import Config, set_seed
from variational_distributions.var_dists import qC, qEpsilonMulti, qT, qMuTau


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

    def test_mutau_init_from_data(self):
        config = Config(n_nodes=4, n_states=5, n_cells=100, chain_length=200, wis_sample_size=100, debug=True)
        data = simulate_full_dataset(config)

        qmt_fixed_init = qMuTau(config).initialize(method='fixed', loc=10., precision_factor=1.,
                                                   shape=5., rate=50.)
        qmt_data_init = qMuTau(config).initialize(method='data', obs=data['obs'])

        print(f"fixed init ELBO: {qmt_fixed_init.compute_elbo()}")
        print(f"data init ELBO: {qmt_data_init.compute_elbo()}")

    def test_true_params_init(self):
        config = Config(n_nodes=5, n_states=7, n_cells=200, chain_length=500, wis_sample_size=20, debug=True)
        joint_q = generate_dataset_var_tree(config)
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
                                                                        "and fixed init lowest elbo")


if __name__ == '__main__':
    unittest.main()
