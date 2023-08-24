import itertools
import unittest
from typing import Tuple, Any

import networkx as nx
import torch
from sklearn.metrics.cluster import adjusted_rand_score

import tests.model_variational_comparisons
import utils.visualization_utils
from variational_distributions.joint_dists import VarTreeJointDist, FixedTreeJointDist
from simul import generate_dataset_var_tree
from tests import model_variational_comparisons
from utils.config import set_seed, Config
from utils.tree_utils import tree_to_newick
from variational_distributions.var_dists import qC, qZ, qMuTau, qPi, qEpsilonMulti, qT, qTauRG


# FIXME: observations wrong shape
@unittest.skip("broken tests")
class updatesRGModelTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # design simple test: fix all other variables
        # and update one var dist alone

        # set all seeds for reproducibility
        set_seed(101)

    def generate_test_dataset_fixed_tree(self, mm: int = 1, step_size: float = 1.) -> tuple[FixedTreeJointDist, Any]:
        """
        Args:
            mm: int. multiplier for longer chain. set it to no more than 10
        Returns:
        """
        # obs with 15 cells, 5 each to different clone
        # in order, clone 0, 1, 2
        cells_per_clone = 100
        chain_length = mm * 10  # total chain length shouldn't be more than 100, ow eps too small
        cfg = Config(n_nodes=3, n_states=5, n_cells=3 * cells_per_clone, chain_length=chain_length, wis_sample_size=2,
                     step_size=step_size, debug=True)
        N = cfg.n_cells
        M = cfg.chain_length
        # obs with 15 cells, 5 each to different clone
        # in order, clone 0, 1, 2
        true_cn_profile = torch.tensor(
            [[2] * 10 * mm,
             [2] * 4 * mm + [3] * 6 * mm,
             [1] * 3 * mm + [3] * 2 * mm + [2] * 3 * mm + [3] * 2 * mm]
            # [3] * 10]
        )
        # cell assignments
        true_z = torch.tensor([0] * cells_per_clone +
                              [1] * cells_per_clone +
                              [2] * cells_per_clone)
        true_pi = torch.nn.functional.one_hot(true_z, num_classes=cfg.n_nodes).float()

        cell_cn_profile = true_cn_profile[true_z, :]
        self.assertEqual(cell_cn_profile.shape, (cfg.n_cells, cfg.chain_length))

        # mean
        alpha_0 = 50
        beta_0 = 5
        tau_dist = torch.distributions.Gamma(alpha_0, beta_0)
        true_tau = tau_dist.sample(torch.tensor([cfg.n_cells]))
        true_gamma = cell_cn_profile.sum(dim=-1)
        R = torch.ones(cfg.n_cells) * 100
        y_dist = torch.distributions.Normal(cell_cn_profile * (R / true_gamma).view(N, 1).expand(N, M),
                                            1. / true_tau.view(N, 1).expand(N, M))
        y = y_dist.sample()
        y = y.T

        true_eps = {
            (0, 1): 1. / (cfg.chain_length - 1),
            (0, 2): 3. / (cfg.chain_length - 1)
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

        fix_qmt = qTauRG(cfg, R, true_params={
            "tau": true_tau
        })
        fix_qmt.gamma = true_gamma

        fix_qpi = qPi(cfg, true_params={
            "pi": torch.ones(cfg.n_nodes) / 3.
        })

        fix_tree = nx.DiGraph()
        fix_tree.add_edges_from([(0, 1), (0, 2)], weight=.5)

        joint_q = FixedTreeJointDist(y, cfg, fix_qc, fix_qz, fix_qeps, fix_qmt, fix_qpi, fix_tree, R)
        return joint_q, true_gamma

    def test_update_qc(self):

        joint_q, gamma = self.generate_test_dataset_fixed_tree(step_size=.1)
        cfg = joint_q.config
        obs = joint_q.obs
        fix_tree = joint_q.T
        fix_qeps = joint_q.eps
        fix_qz = joint_q.z
        fix_qmt = joint_q.mt

        trees = [fix_tree] * cfg.wis_sample_size
        wis_weights = [1 / cfg.wis_sample_size] * cfg.wis_sample_size

        qc = qC(cfg)
        qc.initialize()

        for i in range(50):
            qc.update(obs, fix_qeps, fix_qz, fix_qmt, trees=trees, tree_weights=wis_weights)

        # compare estimated single filtering probs against true copy number profile
        print(joint_q.c)
        print(qc)
        self.assertTrue(torch.all(joint_q.c.true_params["c"] == torch.argmax(qc.single_filtering_probs, dim=-1)))

        self.assertTrue(torch.all(qc.couple_filtering_probs[0, :, 2, 2] > qc.couple_filtering_probs[0, :, 2, 0]))
        self.assertEqual(qc.couple_filtering_probs[1, 3, 2, 3], qc.couple_filtering_probs[1, 3, 2, :].max())
        self.assertEqual(qc.couple_filtering_probs[2, 7, 2, 3], qc.couple_filtering_probs[2, 7, 2, :].max())

    def test_update_qz(self):

        joint_q, gamma = self.generate_test_dataset_fixed_tree()
        cfg = joint_q.config
        obs = joint_q.obs
        fix_qmt = joint_q.mt
        fix_qc = joint_q.c
        fix_qpi = joint_q.pi

        qz = qZ(cfg)
        qz.initialize(z_init='random')

        for i in range(3):
            qz.update(fix_qmt, fix_qc, fix_qpi, obs)

        self.assertTrue(torch.allclose(joint_q.z.true_params["z"],
                                       torch.argmax(qz.exp_assignment(), dim=-1)))

    def test_qtau(self):
        joint_q, gamma = self.generate_test_dataset_fixed_tree()
        cfg = joint_q.config
        obs = joint_q.obs
        fix_qc = joint_q.c
        fix_qz = joint_q.z
        R = joint_q.R
        qtau = qTauRG(cfg, R, alpha_0=50, beta_0=5)
        qtau.initialize()
        # qtau.initialize(method='data', obs=obs)
        for i in range(3):
            qtau.update(fix_qc, fix_qz, obs)

        print(f"qTauRG: {qtau.exp_tau()}")
        print(f"true tau: {joint_q.mt.true_params['tau']}")
        self.assertTrue(torch.allclose(qtau.exp_tau(), joint_q.mt.true_params['tau'], rtol=.2))

    def test_update_qeps(self):

        cfg = Config(n_nodes=5, n_states=7, n_cells=200, chain_length=500, wis_sample_size=20, debug=True)
        joint_q = generate_dataset_var_tree(cfg)
        fix_tree = joint_q.t
        fix_qc = joint_q.c

        qeps = qEpsilonMulti(cfg)
        qeps.initialize('uniform')

        trees, weights = fix_tree.get_trees_sample()

        for i in range(10):
            qeps.update(trees, weights, fix_qc)

        # print(qeps.mean()[[0, 0], [1, 2]])
        true_eps = joint_q.eps.true_params['eps']
        var_eps = qeps.mean()
        self.assertAlmostEqual(var_eps[0, 4],
                               true_eps[0, 4], delta=.1)
        self.assertAlmostEqual(var_eps[4, 3],
                               true_eps[4, 3], delta=.1)
        self.assertAlmostEqual(var_eps[4, 1],
                               true_eps[4, 1], delta=.1)

        self.assertLess(var_eps[1, 2], var_eps[2, 3])

    def test_update_qpi(self):

        joint_q, gamma = self.generate_test_dataset_fixed_tree()
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
        config = Config(n_nodes=5, n_states=7, n_cells=200, chain_length=500, wis_sample_size=20, step_size=.3,
                        debug=True)
        joint_q = generate_dataset_var_tree(config)
        # print(f'obs: {joint_q.obs}')
        # print(f"true c: {joint_q.c.true_params['c']}")
        true_tree_newick = tree_to_newick(joint_q.t.true_params['tree'])
        # print(f"true tree: {true_tree_newick}")
        # print(f"true eps: {joint_q.eps.true_params['eps']}")

        qt = qT(config)
        qt.initialize()

        for i in range(50):
            qt.update(joint_q.c, joint_q.eps)
            qt.get_trees_sample()

        print(qt)
        # sample many trees and get the mode
        n = 500
        k = 10
        trees_sample = qt.get_trees_sample(sample_size=n)
        top_k_trees = utils.tree_utils.top_k_trees_from_sample(*trees_sample, k=k, nx_graph=False)
        self.assertEqual(top_k_trees[0][0], true_tree_newick,
                         msg="true tree is different than top sampled tree by weight\n"
                             f"\t{true_tree_newick} != {top_k_trees[0][0]}:{top_k_trees[0][1]}")
        # print("Sorted trees (by weight sum)")
        # print(top_k_trees)
        top_k_trees = utils.tree_utils.top_k_trees_from_sample(*trees_sample, k=k,
                                                               by_weight=False, nx_graph=False)
        self.assertEqual(top_k_trees[0][0], true_tree_newick,
                         msg="true tree is different than top sampled tree by number of occurrences\n"
                             f"\t{true_tree_newick} != {top_k_trees[0][0]}:{top_k_trees[0][1]}")
        # print("Sorted trees (by occurrence)")
        # print(top_k_trees)

    def test_update_qc_qz(self):

        joint_q, gamma = self.generate_test_dataset_fixed_tree(step_size=0.3)
        cfg = joint_q.config
        obs = joint_q.obs
        fix_tree = joint_q.T
        fix_qpi = joint_q.pi
        fix_qeps = joint_q.eps
        fix_qmt = joint_q.mt
        print(f"Exp log emissions: {fix_qmt.exp_log_emission(obs)[0]}")
        print(f"Exp log emissions: {fix_qmt.exp_log_emission(obs)[6]}")
        print(f"Exp log emissions: {fix_qmt.exp_log_emission(obs)[10]}")
        print(f"Exp log emissions: {fix_qmt.exp_log_emission(obs)[15]}")
        print(f"Exp tau: {fix_qmt.true_params} ")

        qz = qZ(cfg)
        qc = qC(cfg)
        qz.initialize(z_init='random')
        qc.initialize()

        n_iter = 10
        trees = [fix_tree] * cfg.wis_sample_size
        wis_weights = [1 / cfg.wis_sample_size] * cfg.wis_sample_size

        print(f"true c: {joint_q.c.true_params['c']}")
        print(f"true z: {joint_q.z.true_params['z']}")
        for i in range(n_iter):
            qc.update(obs, fix_qeps, qz, fix_qmt, trees=trees, tree_weights=wis_weights)
            qz.update(fix_qmt, qc, fix_qpi, obs)
            print(f"{qz.exp_assignment().mean(dim=0)}")
            print(f"{qc.single_filtering_probs[1].mean(dim=0)}")

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
        model_variational_comparisons.compare_qC_and_true_C(true_C=joint_q.c.true_params["c"], q_c=qc)
        model_variational_comparisons.compare_qZ_and_true_Z(true_Z=joint_q.z.true_params["z"], q_z=qz)

    def test_update_all(self):

        config = Config(n_nodes=5, n_states=7, n_cells=200, chain_length=50, wis_sample_size=30, step_size=.1,
                        debug=True)
        true_joint_q = generate_dataset_var_tree(config)
        joint_q = VarTreeJointDist(config, obs=true_joint_q.obs)
        joint_q.initialize()
        for i in range(50):
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
        print(joint_q)

    def test_update_qz_qmt(self):
        joint_q, gamma = self.generate_test_dataset_fixed_tree()
        cfg = joint_q.config
        obs = joint_q.obs
        fix_qpi = joint_q.pi
        fix_qc = joint_q.c
        n_iter = 40
        cfg.step_size = .3

        almost_true_z_init = joint_q.z.exp_assignment() + .2
        almost_true_z_init /= almost_true_z_init.sum(dim=1, keepdim=True)

        # FIXME: if alpha and beta priors are set to .05, tau estimate is 10x the correct value
        #   undetermined problem lambda * tau ?
        qmt = qMuTau(cfg, nu_prior=1., lambda_prior=.1, alpha_prior=.5, beta_prior=.5)
        # qmt.initialize(method='data', obs=obs)  # does not work
        qmt.initialize(loc=.1, precision_factor=.1, rate=5., shape=5.)
        qz = qZ(cfg).initialize()
        # qz.initialize(method='fixed', pi_init=joint_q.z.exp_assignment() + .2)
        # qz.initialize(method='kmeans', obs=obs)

        print(f"true z: {joint_q.z.true_params['z']}")
        print(f"true tau: {joint_q.mt.true_params['tau']}")
        for i in range(n_iter):
            if i % 5 == 0:
                var_cellassignment = torch.max(qz.pi, dim=-1)[1]
                ari = adjusted_rand_score(joint_q.z.true_params['z'], var_cellassignment)
                print(f"[{i}]")
                print(f"- qz adjusted rand idx: {ari:.2f}")

            qmt.update(fix_qc, qz, obs)
            qz.update(qmt)
        print(f"results after {n_iter} iter")
        print(f"- var z: {torch.max(qz.pi, dim=-1)[1]}")
        print(f"- var tau: {qmt.exp_tau()}")

    def test_update_qc_qz_qmt(self):
        joint_q, gamma = self.generate_test_dataset_fixed_tree()
        cfg = joint_q.config
        obs = joint_q.obs
        fix_tree = joint_q.T
        fix_qpi = joint_q.pi
        fix_qeps = joint_q.eps

        R = joint_q.mt.R
        qmt = qTauRG(cfg, R, alpha_0=25., beta_0=5.)
        qz = qZ(cfg)
        qc = qC(cfg)
        almost_true_z_init = joint_q.z.exp_assignment() + .2
        almost_true_z_init /= almost_true_z_init.sum(dim=1, keepdim=True)
        qmt.initialize()
        qz.initialize(z_init='fixed', pi_init=joint_q.z.exp_assignment() + .2)
        # qz.initialize(method='kmeans', obs=obs)
        qc.initialize()

        trees = [fix_tree] * cfg.wis_sample_size
        wis_weights = [1 / cfg.wis_sample_size] * cfg.wis_sample_size

        # change step_size
        cfg.step_size = .3
        # print(obs)

        utils.visualization_utils.visualize_copy_number_profiles(joint_q.c.true_params['c'],
                                                                 save_path="../test_output/update_qcqzqmt_true_cn.png",
                                                                 title_suff="- true values")
        for i in range(20):
            if i % 5 == 0:
                # print(f"Iter {i} qZ: {qz.exp_assignment()}")
                # print(f"iter {i} qmt mean for each cell: {qmt.nu}")
                # print(f"iter {i} qmt tau for each cell: {qmt.exp_tau()}")
                partial_elbo = qc.compute_elbo([fix_tree], [1.], fix_qeps) + qz.compute_elbo(fix_qpi)
                utils.visualization_utils.visualize_copy_number_profiles(
                    torch.argmax(qc.single_filtering_probs, dim=-1),
                    save_path=f"./test_output/update_qcqzqmt_it{i}_var_cn.png", title_suff=f"- VI iter {i},"
                                                                                           f" elbo: {partial_elbo}")
            qmt.update(qc, qz, obs)
            qz.update(qmt, qc, fix_qpi, obs)
            qc.update(obs, fix_qeps, qz, qmt, trees=trees, tree_weights=wis_weights)

        # print(qmt.exp_tau())
        # print(joint_q.mt.true_params['tau'])
        # print(joint_q.z.true_params["z"])
        # print(torch.argmax(qz.exp_assignment(), dim=-1))
        # self.assertTrue(torch.allclose(joint_q.z.true_params["z"],
        #                                torch.argmax(qz.exp_assignment(), dim=-1)))
        var_cellassignment = torch.max(qz.pi, dim=-1)[1]
        ari = adjusted_rand_score(joint_q.z.true_params['z'], var_cellassignment)
        self.assertGreater(ari, .85)
        self.assertTrue(torch.all(joint_q.c.true_params["c"] == torch.argmax(qc.single_filtering_probs, dim=-1)))
        tests.model_variational_comparisons.fixed_T_comparisons(obs, true_C=joint_q.c.true_params["c"],
                                                                true_Z=joint_q.z.true_params["z"], true_pi=None,
                                                                true_mu=joint_q.mt.true_params["mu"],
                                                                true_tau=joint_q.mt.true_params["tau"],
                                                                true_epsilon=None, q_c=qc, q_z=qz, qpi=None, q_mt=qmt)

    def test_label_switching(self):
        # define 3 clones besides root
        cfg = Config(n_nodes=4, n_states=5, n_cells=40, chain_length=50, wis_sample_size=2, step_size=1, debug=True)
        true_cn_profile = torch.tensor(
            [[2] * cfg.chain_length,
             [2] * 30 + [3] * 20,
             [1] * 10 + [3] * 25 + [2] * 10 + [4] * 5,
             [3] * 10 + [4] * 30 + [1] * 10]
        )

        # cell assignments
        true_z = torch.tensor([0] * 5 +
                              [1] * 10 +
                              [2] * 7 +
                              [3] * 18)

        true_pi = torch.tensor([5, 10, 7, 18]) / cfg.n_cells
        self.assertEqual(true_pi.sum(), 1.)

        cell_cn_profile = true_cn_profile[true_z, :]
        self.assertEqual(cell_cn_profile.shape, (cfg.n_cells, cfg.chain_length))

        # mean and precision
        tau = torch.tensor(5)
        nu, lmbda = torch.tensor([1, 5])  # randomize mu for each cell with these hyperparameters
        true_mu = torch.randn(cfg.n_cells) / torch.sqrt(lmbda * tau) + nu
        obs = (cell_cn_profile * true_mu[:, None]).T.clamp(min=0)
        self.assertEqual(obs.shape, (cfg.chain_length, cfg.n_cells))
        print(obs)

        # initialize main dists
        # qz = qZ(cfg).initialize(method='kmeans', obs=obs)
        # skewed towards true cluster, but not exact
        qz = qZ(cfg).initialize(z_init='fixed',
                                pi_init=torch.nn.functional.one_hot(true_z).float().clamp(.2 / (cfg.n_nodes - 1), .8))
        # qc = qC(cfg).initialize(method='bw-cluster', obs=obs, clusters=qz.kmeans_labels)
        qc = qC(cfg).initialize()
        qmt = qMuTau(cfg).initialize(method='data', obs=obs)
        qeps = qEpsilonMulti(cfg).initialize()
        qpi = qPi(cfg).initialize()
        qt = qT(cfg).initialize()
        joint_q = VarTreeJointDist(cfg, qc=qc, qz=qz, qmt=qmt, qeps=qeps, qpi=qpi, qt=qt, obs=obs)

        # update and check copy numbers
        print(f"[init] elbo: {joint_q.compute_elbo(*joint_q.t.get_trees_sample(sample_size=10))}")
        utils.visualization_utils.visualize_copy_number_profiles(true_cn_profile)
        print(f"true z: {true_z}")
        print(f"[init] var z: {qz.exp_assignment()}")
        for i in range(30):
            joint_q.update()
            if i % 5 == 0:
                utils.visualization_utils.visualize_copy_number_profiles(
                    torch.argmax(joint_q.c.single_filtering_probs, dim=-1))
                print(f"[{i}] var z: {qz.exp_assignment()}")

                trees, weights = joint_q.t.get_trees_sample(sample_size=10)
                print(f"[{i}] elbo: {joint_q.compute_elbo(trees, weights)}")

        # same but with z at current pos


if __name__ == '__main__':
    unittest.main()
