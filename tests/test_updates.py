import unittest

import networkx as nx
import torch

import simul
from inference.copy_tree import VarDistFixedTree, JointVarDist
from simul import tree_to_newick
from utils.config import set_seed, Config
from variational_distributions.var_dists import qC, qZ, qMuTau, qPi, qEpsilonMulti, qT
from tests.utils_testing import simul_data_pyro_full_model


class updatesTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # design simple test: fix all other variables
        # and update one var dist alone

        # set all seeds for reproducibility
        set_seed(101)

    def generate_test_dataset_pyro(self, config: Config):
        tree = nx.random_tree(config.n_nodes, create_using=nx.DiGraph)
        fict_data = torch.ones((config.chain_length, config.n_cells))
        data = simul_data_pyro_full_model(fict_data, config.n_cells,
                                          config.chain_length,
                                          config.n_states, tree,
                                          mu_0=1., lambda_0=10.)
        return data + (tree, )




    def generate_test_dataset_fixed_tree(self) -> VarDistFixedTree:
        # obs with 15 cells, 5 each to different clone
        # in order, clone 0, 1, 2
        cells_per_clone = 10
        mm = 1  # change this to increase length
        chain_length = mm * 10  # total chain length shouldn't be more than 100, ow eps too small
        cfg = Config(n_nodes=3, n_states=5, n_cells=3 * cells_per_clone, chain_length=chain_length,
                     wis_sample_size=2, debug=True)
        # obs with 15 cells, 5 each to different clone
        # in order, clone 0, 1, 2
        true_cn_profile = torch.tensor(
            [[2] * 10*mm,
             [2] * 4*mm + [3] * 6*mm,
             [1] * 3*mm + [3] * 2*mm + [2] * 3*mm + [3] * 2*mm]
            # [3] * 10]
        )
        # cell assignments
        true_z = torch.tensor([0] * cells_per_clone +
                              [1] * cells_per_clone +
                              [2] * cells_per_clone)
        true_pi = torch.nn.functional.one_hot(true_z, num_classes=cfg.n_nodes).float()

        cell_cn_profile = true_cn_profile[true_z, :]
        self.assertEqual(cell_cn_profile.shape, (cfg.n_cells, cfg.chain_length))

        # mean and precision
        nu, lmbda = torch.tensor([1, 10])  # randomize mu for each cell with these hyperparameters
        true_mu = torch.randn(cfg.n_cells) / torch.sqrt(lmbda) + nu
        obs = (cell_cn_profile * true_mu[:, None]).T.clamp(min=0)
        self.assertEqual(obs.shape, (cfg.chain_length, cfg.n_cells))

        true_eps = torch.ones((cfg.n_nodes, cfg.n_nodes))
        true_eps[0, 1] = 1./(cfg.chain_length-1)
        true_eps[0, 2] = 3./(cfg.chain_length-1)

        # give true values to the other required dists
        fix_qc = qC(cfg, true_params={
            "c": true_cn_profile
        })

        fix_qz = qZ(cfg, true_params={
            "z": true_z
        })

        fix_qeps = qEpsilonMulti(cfg, true_params={
            "eps": true_eps
        })

        fix_qmt = qMuTau(cfg, true_params={
            "mu": true_mu,
            "tau": torch.ones(cfg.n_cells) * lmbda
        })

        fix_qpi = qPi(cfg, true_params={
            "pi": torch.ones(cfg.n_nodes) / 3.
        })

        fix_tree = nx.DiGraph()
        fix_tree.add_edges_from([(0, 1), (0, 2)], weight=.5)

        joint_q = VarDistFixedTree(cfg, fix_qc, fix_qz, fix_qeps,
                                   fix_qmt, fix_qpi, fix_tree, obs)
        return joint_q


    def generate_test_dataset_var_tree(self, config: Config) -> JointVarDist:
        simul_data = simul.simulate_full_dataset(config, eps_a=2, eps_b=5)

        fix_qc = qC(config, true_params={
            "c": simul_data['c']
        })

        fix_qz = qZ(config, true_params={
            "z": simul_data['z']
        })

        fix_qeps = qEpsilonMulti(config, true_params={
            "eps": simul_data['eps']
        })

        fix_qmt = qMuTau(config, true_params={
            "mu": simul_data['mu'],
            "tau": simul_data['tau']
        })

        fix_qpi = qPi(config, true_params={
            "pi": simul_data['pi']
        })

        fix_qt = qT(config, true_params={
            "tree": simul_data['tree']
        })

        joint_q = JointVarDist(config, fix_qc, fix_qz, fix_qt, fix_qeps,
                               fix_qmt, fix_qpi, simul_data['obs'])
        return joint_q


    def test_update_qt_simul_data(self):
        config = Config(n_nodes=4, n_states=5, eps0=1e-2, n_cells=20, chain_length=20, wis_sample_size=10,
                        debug=True)
        c, obs, z, pi, mu, tau, eps, tree = self.generate_test_dataset_pyro(config)
        print(obs)
        print(c)
        print(tree_to_newick(tree))


    def test_update_qt(self):

        joint_q = self.generate_test_dataset_fixed_tree()
        cfg = joint_q.config
        fix_tree = joint_q.T
        fix_qeps = joint_q.eps
        fix_qc = joint_q.c

        trees = [fix_tree] * cfg.wis_sample_size

        qt = qT(cfg)
        qt.initialize()

        for i in range(100):
            trees_sample, iw = qt.get_trees_sample(sample_size=cfg.wis_sample_size)
            qt.update(trees_sample, fix_qc, fix_qeps)

        print(qt.weighted_graph.edges.data())

    def test_update_qc(self):

        joint_q = self.generate_test_dataset_fixed_tree()
        cfg = joint_q.config
        obs = joint_q.obs
        fix_tree = joint_q.T
        fix_qeps = joint_q.eps
        fix_qz = joint_q.z
        fix_qmt = joint_q.mt

        trees = [fix_tree] * cfg.wis_sample_size
        wis_weights = [1/cfg.wis_sample_size] * cfg.wis_sample_size

        qc = qC(cfg)
        qc.initialize()

        for i in range(100):
            qc.update(obs, fix_qeps, fix_qz, fix_qmt,
                      trees=trees, tree_weights=wis_weights)

        # compare estimated single filtering probs against true copy number profile
        self.assertTrue(torch.all(joint_q.c.true_params["c"] == torch.argmax(qc.single_filtering_probs, dim=-1)))

        self.assertTrue(torch.all(qc.couple_filtering_probs[0, :, 2, 2] > qc.couple_filtering_probs[0, :, 2, 0]))
        self.assertEqual(qc.couple_filtering_probs[1, 3, 2, 3], qc.couple_filtering_probs[1, 3, 2, :].max())
        self.assertEqual(qc.couple_filtering_probs[2, 7, 2, 3], qc.couple_filtering_probs[2, 7, 2, :].max())

    def test_update_qz(self):

        joint_q = self.generate_test_dataset_fixed_tree()
        cfg = joint_q.config
        obs = joint_q.obs
        fix_qmt = joint_q.mt
        fix_qc = joint_q.c
        fix_qpi = joint_q.pi

        qz = qZ(cfg)
        qz.initialize(method='random')

        for i in range(3):
            qz.update(fix_qmt, fix_qc, fix_qpi, obs)

        self.assertTrue(torch.allclose(joint_q.z.true_params["z"],
                                       torch.argmax(qz.exp_assignment(), dim=-1)))

    def test_qmt(self):

        joint_q = self.generate_test_dataset_fixed_tree()
        cfg = joint_q.config
        obs = joint_q.obs
        fix_qc = joint_q.c
        fix_qz = joint_q.z

        qmt = qMuTau(cfg)
        # uninformative initialization of mu0, tau0, alpha0, beta0
        qmt.initialize(loc=0, precision_factor=.1, rate=.5, shape=.5)
        for i in range(10):
            qmt.update(fix_qc, fix_qz, obs)

        # print(qmt.exp_tau())
        # print(joint_q.mt.true_params['tau'])
        self.assertTrue(torch.allclose(qmt.nu, joint_q.mt.true_params['mu'], rtol=1e-2))
        self.assertTrue(torch.allclose(qmt.exp_tau(), joint_q.mt.true_params['tau'], rtol=.2))

    def test_qeps(self):

        joint_q = self.generate_test_dataset_fixed_tree()
        cfg = joint_q.config
        fix_tree = joint_q.T
        fix_qc = joint_q.c

        qeps = qEpsilonMulti(cfg)
        qeps.initialize('uniform')

        trees = [fix_tree] * cfg.wis_sample_size
        wis_weights = [1/cfg.wis_sample_size] * cfg.wis_sample_size

        for i in range(10):
            qeps.update(trees, wis_weights, fix_qc.couple_filtering_probs)

        # print(qeps.mean()[[0, 0], [1, 2]])
        true_eps = joint_q.eps.true_params['eps']
        var_eps = qeps.mean()
        self.assertAlmostEqual(var_eps[0, 1],
                               true_eps[0, 1], delta=.1)
        self.assertAlmostEqual(var_eps[0, 2],
                               true_eps[0, 2], delta=.1)

        self.assertGreater(var_eps[0, 2], var_eps[0, 1])

    def test_update_qpi(self):

        joint_q = self.generate_test_dataset_fixed_tree()
        cfg = joint_q.config
        fix_qz = joint_q.z

        qpi = qPi(cfg)
        qpi.initialize('random')
        # print(f'init exp pi: {qpi.exp_log_pi().exp()}')

        n_iter = 100
        for i in range(n_iter):
            qpi.update(fix_qz)

        # print(f'after {n_iter} iter - exp pi: {qpi.exp_log_pi().exp()}')
        # print(f'true exp pi: {joint_q.pi.exp_log_pi().exp()}')
        self.assertTrue(torch.allclose(qpi.exp_log_pi().exp(),
                                       joint_q.pi.exp_log_pi().exp(), rtol=1e-2))

    def test_update_qc_qz(self):

        joint_q = self.generate_test_dataset_fixed_tree()
        cfg = joint_q.config
        obs = joint_q.obs
        fix_tree = joint_q.T
        fix_qpi = joint_q.pi
        fix_qeps = joint_q.eps
        fix_qmt = joint_q.mt

        qz = qZ(cfg)
        qc = qC(cfg)
        qz.initialize(method='random')
        qc.initialize()

        trees = [fix_tree] * cfg.wis_sample_size
        wis_weights = [1/cfg.wis_sample_size] * cfg.wis_sample_size

        for i in range(10):
            qc.update(obs, fix_qeps, qz, fix_qmt,
                      trees=trees, tree_weights=wis_weights)
            qz.update(fix_qmt, qc, fix_qpi, obs)

        self.assertTrue(torch.allclose(joint_q.z.true_params["z"],
                                       torch.argmax(qz.exp_assignment(), dim=-1)))

        self.assertTrue(torch.all(joint_q.c.true_params["c"] == torch.argmax(qc.single_filtering_probs, dim=-1)))

    def test_update_qc_qz_qmt(self):

        joint_q = self.generate_test_dataset_fixed_tree()
        cfg = joint_q.config
        obs = joint_q.obs
        fix_tree = joint_q.T
        fix_qpi = joint_q.pi
        fix_qeps = joint_q.eps

        qmt = qMuTau(cfg)
        qz = qZ(cfg)
        qc = qC(cfg)
        qmt.initialize(loc=0, precision_factor=.1, rate=.5, shape=.5)
        qz.initialize(method='random')
        qc.initialize()

        trees = [fix_tree] * cfg.wis_sample_size
        wis_weights = [1/cfg.wis_sample_size] * cfg.wis_sample_size

        # change step_size
        cfg.step_size = .2

        for i in range(100):
            qc.update(obs, fix_qeps, qz, qmt,
                      trees=trees, tree_weights=wis_weights)
            qz.update(qmt, qc, fix_qpi, obs)
            print(qz.exp_assignment())
            qmt.update(qc, qz, obs)

        # print(qmt.exp_tau())
        # print(joint_q.mt.true_params['tau'])
        self.assertTrue(torch.allclose(joint_q.z.true_params["z"],
                                       torch.argmax(qz.exp_assignment(), dim=-1)))

        self.assertTrue(torch.all(joint_q.c.true_params["c"] == torch.argmax(qc.single_filtering_probs, dim=-1)))

if __name__ == '__main__':
    unittest.main()
