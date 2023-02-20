import itertools
import unittest

import networkx as nx
import torch
from sklearn.metrics.cluster import adjusted_rand_score

import simul
from inference.copy_tree import VarDistFixedTree, JointVarDist
from utils.config import set_seed, Config
from utils.tree_utils import tree_to_newick
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




    def generate_test_dataset_fixed_tree(self, mm: int = 1, step_size: float = 1.) -> VarDistFixedTree:
        """
        Args:
            mm: int. multiplier for longer chain. set it to no more than 10
        Returns:
        """
        # obs with 15 cells, 5 each to different clone
        # in order, clone 0, 1, 2
        cells_per_clone = 10
        chain_length = mm * 10  # total chain length shouldn't be more than 100, ow eps too small
        cfg = Config(n_nodes=3, n_states=5, n_cells=3 * cells_per_clone, chain_length=chain_length,
                     wis_sample_size=2, debug=True, step_size=step_size)
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

        true_eps = {
            (0, 1): 1./(cfg.chain_length-1),
            (0, 2): 3./(cfg.chain_length-1)
        }

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


    def generate_dataset_var_tree(self, config: Config) -> JointVarDist:
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

        joint_q = JointVarDist(config, simul_data['obs'], fix_qc, fix_qz, fix_qt, fix_qeps, fix_qmt, fix_qpi)
        return joint_q

    def test_update_qt_simul_data(self):
        config = Config(n_nodes=4, n_states=5, eps0=1e-2, n_cells=100, chain_length=20, wis_sample_size=10,
                        debug=True)
        joint_q = self.generate_dataset_var_tree(config)
        print(f'obs: {joint_q.obs}')
        print(f"true c: {joint_q.c.true_params['c']}")
        print(f"true tree: {tree_to_newick(joint_q.t.true_params['tree'])}")
        print(f"true eps: {joint_q.eps.true_params['eps']}")

        qt = qT(config)
        qt.initialize()

        for i in range(10):
            trees_sample, weights = qt.get_trees_sample()
            qt.update(joint_q.c, joint_q.eps)
            print(qt.elbo())
            for t, w in zip(trees_sample, weights):
                print(f"{tree_to_newick(t)} | {w}")

        print(qt.weighted_graph.edges.data())


    def test_update_qt(self):

        joint_q = self.generate_test_dataset_fixed_tree(mm=10)
        cfg = joint_q.config
        fix_tree = joint_q.T
        fix_qeps = joint_q.eps
        fix_qc = joint_q.c

        qt = qT(cfg)
        qt.initialize()

        print(tree_to_newick(fix_tree, weight='weight'))
        for i in range(100):
            trees_sample, weights = qt.get_trees_sample()
            qt.update(fix_qc, fix_qeps)
            for t, w in zip(trees_sample, weights):
                print(f"{tree_to_newick(t)} | {w}")

        # print(qt.weighted_graph.edges.data())
        # sample_size = 20
        # sampled_trees, sampled_weights = qt.get_trees_sample(sample_size=sample_size)
        # for t, w in zip(sampled_trees, sampled_weights):
        #     print(f"{tree_to_newick(t)} | {w}")

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
        qc.initialize(method='bw-cluster', obs=obs, clusters=fix_qz.true_params['z'])

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
        # qmt.initialize(method='data', obs=obs)
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
            qeps.update(trees, wis_weights, fix_qc)

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

    def test_update_large_qt(self):
        config = Config(n_nodes=5, n_states=7, n_cells=200, chain_length=500,
                        wis_sample_size=20, debug=True, step_size=.3)
        joint_q = self.generate_dataset_var_tree(config)
        print(f'obs: {joint_q.obs}')
        print(f"true c: {joint_q.c.true_params['c']}")
        print(f"true tree: {tree_to_newick(joint_q.t.true_params['tree'])}")
        print(f"true eps: {joint_q.eps.true_params['eps']}")

        qt = qT(config)
        qt.initialize()

        for i in range(50):
            qt.update(joint_q.c, joint_q.eps)
            qt.get_trees_sample()

        print(sorted(qt.weighted_graph.edges.data(), key=lambda e: e[2]['weight'], reverse=True))

    def test_update_qc_qz(self):

        joint_q = self.generate_test_dataset_fixed_tree(step_size=0.3)
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

        n_iter = 10
        trees = [fix_tree] * cfg.wis_sample_size
        wis_weights = [1/cfg.wis_sample_size] * cfg.wis_sample_size

        print(f"true c: {joint_q.c.true_params['c']}")
        print(f"true z: {joint_q.z.true_params['z']}")
        for i in range(n_iter):
            qc.update(obs, fix_qeps, qz, fix_qmt,
                      trees=trees, tree_weights=wis_weights)
            qz.update(fix_qmt, qc, fix_qpi, obs)

        print(f"after {n_iter} iterations")
        print("var cell assignments")
        var_cellassignment = torch.max(qz.pi, dim=-1)[1]
        print(var_cellassignment)
        print("var copy number")
        var_copynumber = torch.max(qc.single_filtering_probs, dim=-1)[1]
        print(var_copynumber)

        # assignment can differ in the labels, using adjusted rand index to
        # evaluate the variational qz inference
        ari = adjusted_rand_score(joint_q.z.true_params['z'], var_cellassignment)
        print(f"ARI={ari}")
        self.assertGreater(ari, .9)
        perms = list(itertools.permutations(range(cfg.n_nodes)))
        # compare with all permutations of copy numbers
        # take the maximum match according to accuracy ratio
        c_accuracy = torch.max((var_copynumber[perms, :] ==
                                joint_q.c.true_params["c"]).sum(2).sum(1) / (cfg.n_nodes * cfg.chain_length))
        print(f"copy number accuracy: {c_accuracy}")
        self.assertGreater(c_accuracy, .9)

    def test_update_all(self):

        config = Config(n_nodes=5, n_states=7, n_cells=200, chain_length=50,
                        wis_sample_size=20, debug=True, step_size=.3)
        true_joint_q = self.generate_dataset_var_tree(config)
        joint_q = JointVarDist(config, obs=true_joint_q.obs)
        joint_q.initialize()
        for i in range(20):
            joint_q.update()

        print(f'true c at node 1: {true_joint_q.c.single_filtering_probs[1].max(dim=-1)[1]}')
        print(f'var c at node 1: {joint_q.c.single_filtering_probs[1].max(dim=-1)[1]}')

        print(f'true tree: {tree_to_newick(true_joint_q.t.true_params["tree"])}')
        print(f'var tree graph: '
              f'{sorted(joint_q.t.weighted_graph.edges.data("weight"), key=lambda e: e[2], reverse=True)}')
        sample_size = 100
        s_trees, s_weights = joint_q.t.get_trees_sample(sample_size=sample_size)
        accum = {}
        for t, w in zip(s_trees, s_weights):
            tnwk = tree_to_newick(t)
            if tnwk in accum:
                accum[tnwk] += w
            else:
                accum[tnwk] = w

        print(sorted(accum.items(), key=lambda x: x[1], reverse=True))
        # NOTE: copy number is not very accurate and tree sampling is not exact, but still some
        #   of the true edges obtain high probability of being sampled.
        #   also, the weights don't explode to very large or very small values, causing the algorithm to crash


    def test_update_qc_qz_qmt(self):

        # FIXME: try other initialization strategies
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
        print(obs)

        for i in range(20):
            qz.update(qmt, qc, fix_qpi, obs)
            qmt.update(qc, qz, obs)
            qc.update(obs, fix_qeps, qz, qmt,
                      trees=trees, tree_weights=wis_weights)
            print(f"Iter {i} qZ mean: {qz.exp_assignment().mean(dim=0)}")
            print(f"iter {i} qmt mean for each cell: {qmt.nu}")

        # print(qmt.exp_tau())
        # print(joint_q.mt.true_params['tau'])
        print(joint_q.z.true_params["z"])
        print(torch.argmax(qz.exp_assignment(), dim=-1))
        # self.assertTrue(torch.allclose(joint_q.z.true_params["z"],
                                       #torch.argmax(qz.exp_assignment(), dim=-1)))

        self.assertTrue(torch.all(joint_q.c.true_params["c"] == torch.argmax(qc.single_filtering_probs, dim=-1)))

if __name__ == '__main__':
    unittest.main()
