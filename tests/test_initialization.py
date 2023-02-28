import unittest

from simul import simulate_full_dataset
from utils.config import Config, set_seed
from variational_distributions.var_dists import qC, qEpsilonMulti, qT


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
        rand_elbo = qc_rand.elbo(trees_sample, trees_weights, fix_qeps)

        # Baum-Welch init
        qc_bw = qC(config).initialize(method='bw-cluster', obs=data['obs'], clusters=data['z'])
        bw_elbo = qc_bw.elbo(trees_sample, trees_weights, fix_qeps)

        self.assertGreater(bw_elbo, rand_elbo)

        fix_qc = qC(config, true_params={
            "c": data['c']
        })


if __name__ == '__main__':
    unittest.main()