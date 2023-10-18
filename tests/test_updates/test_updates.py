import itertools
import unittest

import networkx as nx
import numpy as np
import torch
from sklearn.metrics import v_measure_score
from sklearn.metrics.cluster import adjusted_rand_score

import matplotlib
import utils.visualization_utils
from experiments.fixed_tree_experiments.k4_prior_gridsearch import sample_dataset_generation
from inference.victree import make_input, VICTree
from utils import tree_utils
from utils.evaluation import evaluate_victree_to_df, best_mapping
from variational_distributions.joint_dists import VarTreeJointDist, FixedTreeJointDist
from simul import generate_dataset_var_tree
from tests import model_variational_comparisons
from utils.config import set_seed, Config
from utils.tree_utils import tree_to_newick
from variational_distributions.var_dists import qC, qZ, qMuTau, qPi, qEpsilonMulti, qT


class updatesTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # design simple test: fix all other variables
        # and update one var dist alone

        # set all seeds for reproducibility
        set_seed(101)

    def generate_test_dataset_fixed_tree(self, mm: int = 1, step_size: float = 1.) -> FixedTreeJointDist:
        """
        Args:
            mm: int. multiplier for longer chain. set it to no more than 10
        Returns:
        """
        # in order, clone 0, 1, 2
        cells_per_clone = 10
        chain_length = mm * 10  # total chain length shouldn't be more than 100, ow eps too small
        cfg = Config(n_nodes=3, n_states=5, n_cells=3 * cells_per_clone, chain_length=chain_length, wis_sample_size=2,
                     step_size=step_size, debug=True)
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

        # mean and precision
        nu, lmbda = torch.tensor([1, 10])  # randomize mu for each cell with these hyperparameters
        tau = 10
        true_mu = torch.randn(cfg.n_cells) / torch.sqrt(lmbda * tau) + nu
        obs = (cell_cn_profile * true_mu[:, None]).T.clamp(min=0)
        self.assertEqual(obs.shape, (cfg.chain_length, cfg.n_cells))

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

        fix_qmt = qMuTau(cfg, true_params={
            "mu": true_mu,
            "tau": torch.ones(cfg.n_cells) * lmbda
        })

        fix_qpi = qPi(cfg, true_params={
            "pi": torch.ones(cfg.n_nodes) / 3.
        })

        fix_tree = nx.DiGraph()
        fix_tree.add_edges_from([(0, 1), (0, 2)], weight=.5)

        joint_q = FixedTreeJointDist(obs, cfg, fix_qc, fix_qz, fix_qeps, fix_qmt, fix_qpi, fix_tree)
        return joint_q

    def test_update_qt_simul_data(self):
        config = Config(n_nodes=4, n_states=5, eps0=1e-2, n_cells=100, chain_length=20, wis_sample_size=10, debug=True)
        joint_q = generate_dataset_var_tree(config)
        print(f'obs: {joint_q.obs}')
        print(f"true c: {joint_q.c.true_params['c']}")
        print(f"true tree: {tree_to_newick(joint_q.t.true_params['tree'])}")
        print(f"true eps: {joint_q.eps.true_params['eps']}")

        qt = qT(config)
        qt.initialize()

        for i in range(10):
            trees_sample, weights = qt.get_trees_sample()
            qt.update(joint_q.c, joint_q.eps)
            eval_trees_sample, eval_weights = qt.get_trees_sample()
            print(qt.compute_elbo(eval_trees_sample, eval_weights))
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

        print(qT(cfg, true_params={'tree': fix_tree}))
        print(qt)

        # print(qt.weighted_graph.edges.data())
        # sample_size = 20
        # sampled_trees, sampled_weights = qt.get_trees_sample(sample_size=sample_size)
        # for t, w in zip(sampled_trees, sampled_weights):
        #     print(f"{tree_to_newick(t)} | {w}")

    def test_update_qc(self):

        joint_q = self.generate_test_dataset_fixed_tree(step_size=.1)
        cfg = joint_q.config
        obs = joint_q.obs
        fix_tree = joint_q.T
        fix_qeps = joint_q.eps
        fix_qz = joint_q.z
        fix_qmt = joint_q.mt

        trees = [fix_tree] * cfg.wis_sample_size
        wis_weights = [1 / cfg.wis_sample_size] * cfg.wis_sample_size

        qc = qC(cfg)
        # qc.initialize(method='bw-cluster', obs=obs, clusters=fix_qz.true_params['z'])
        qc.initialize(method='random')

        for i in range(50):
            qc.update(obs, fix_qeps, fix_qz, fix_qmt, trees=trees, tree_weights=wis_weights)

        # compare estimated single filtering probs against true copy number profile
        print(qc)
        # print(joint_q.c)
        # print(qc.get_viterbi())
        self.assertTrue(torch.all(joint_q.c.true_params['c'] == qc.get_viterbi()), "true c does not match viterbi path")

        self.assertTrue(torch.all(joint_q.c.true_params["c"] == torch.argmax(qc.single_filtering_probs, dim=-1)),
                        "true c does not match argmax path")

        self.assertTrue(torch.all(qc.couple_filtering_probs[0, :, 2, 2] > qc.couple_filtering_probs[0, :, 2, 0]))
        self.assertEqual(qc.couple_filtering_probs[1, 3, 2, 3], qc.couple_filtering_probs[1, 3, 2, :].max())
        self.assertEqual(qc.couple_filtering_probs[2, 7, 2, 3], qc.couple_filtering_probs[2, 7, 2, :].max())

    def test_update_qc_simul_data(self):
        max_iter = 50
        rtol = 1e-3
        config = Config(n_nodes=5, n_cells=300, chain_length=1000, step_size=.2, debug=True)
        joint_q = generate_dataset_var_tree(config, eps_a=10., eps_b=4000.,
                                            nu_prior=1., lambda_prior=3., alpha_prior=2500., beta_prior=50.,
                                            cne_length_factor=200, dir_alpha=10.)
        true_tree = joint_q.t.true_params['tree']
        print(f"true tree: {tree_to_newick(true_tree)}")

        qc = qC(config)
        qc.initialize(method='diploid')

        i = 0
        convergence = False
        curr_elbo = - np.inf
        while not convergence and i < max_iter:
            qc.update(joint_q.obs, joint_q.eps, joint_q.z, joint_q.mt, [true_tree], [1.])
            new_elbo = qc.compute_elbo([true_tree], [1.], joint_q.eps)
            improvement = abs((new_elbo - curr_elbo) / curr_elbo)
            print(f"[{i}] elbo: {new_elbo:.3f} (rel impr: {improvement:.3f})")
            if improvement < rtol:
                convergence = True
                print(f"converged after {i} iterations")
            curr_elbo = new_elbo
            i += 1

        # mean abs deviation
        mad = torch.mean(torch.abs(qc.get_viterbi() - joint_q.c.true_params['c']).float())
        print(f"MAD: {mad}")
        self.assertLess(mad, 0.1)

    def test_update_qz(self):

        joint_q = self.generate_test_dataset_fixed_tree()
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

    def test_qmt(self):

        joint_q, adata = sample_dataset_generation(seed=0)
        cfg = joint_q.config
        obs = joint_q.obs
        fix_qc = joint_q.c
        fix_qz = joint_q.z

        qmt = qMuTau(cfg, from_obs=(obs, 1.))
        # initialization of mu0, tau0, alpha0, beta0 which scales with data size
        qmt.initialize(method='data-size', obs=obs)
        for i in range(20):
            qmt.update(fix_qc, fix_qz, obs)

        # print(qmt.exp_tau())
        # print(joint_q.mt.true_params['tau'])
        mu_mse = torch.mean(torch.pow(qmt.nu - joint_q.mt.true_params['mu'], torch.tensor(2))).item()
        self.assertAlmostEqual(mu_mse, 0., places=3)
        # compute the mean of the pdf computed in each tau value under the estimated alpha/gamma params
        # as one way of evaluating tau correctness
        tau_log_p = torch.distributions.Gamma(qmt.alpha, qmt.beta,
                                              validate_args=True).log_prob(joint_q.mt.true_params['tau'])
        mean_tau_pdf = torch.exp(tau_log_p).mean().item()
        self.assertGreater(mean_tau_pdf, 0.7, msg="tau is not properly estimated")

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

        joint_q = self.generate_test_dataset_fixed_tree()
        cfg = joint_q.config
        fix_qz = joint_q.z

        qpi = qPi(cfg)
        qpi.initialize('random')
        # print(f'init exp pi: {qpi.exp_log_pi().exp()}')

        n_iter = 100
        for i in range(n_iter):
            qpi.update(fix_qz)

        self.assertTrue(torch.allclose(qpi.exp_pi(),
                                       joint_q.pi.true_params['pi']))

    def test_unbalanced_qpi(self):
        config = Config(n_nodes=5, n_cells=300,
                        debug=True)
        joint_q = generate_dataset_var_tree(config, dir_alpha=[100., 300., 1000., 200., 500.])
        fix_qz = joint_q.z

        qpi = qPi(config, delta_prior=1).initialize('random')

        n_iter = 10
        for i in range(n_iter):
            qpi.update(fix_qz)
        # compare with the actual ratio of cell assigned to clones in the true z
        target_pi = torch.bincount(joint_q.z.true_params['z']) / config.n_cells
        vi_pi = qpi.exp_pi()

        print(vi_pi, target_pi)
        self.assertTrue(torch.allclose(vi_pi,
                                       target_pi, atol=0.005))

    def test_update_large_qt(self):
        config = Config(n_nodes=5, n_states=7, n_cells=200, chain_length=500, wis_sample_size=20, step_size=.3,
                        debug=True)
        joint_q = generate_dataset_var_tree(config)
        # print(f'obs: {joint_q.obs}')
        # print(f"true c: {joint_q.c.true_params['c']}")
        true_tree_newick = tree_to_newick(joint_q.t.true_params['tree'])
        print(f"true tree: {true_tree_newick}")
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

    @unittest.skip('long exec time')
    def test_full_updates(self):
        k = 6
        seed = 101
        true_jq, adata = sample_dataset_generation(K=k, seed=seed)
        config, jq, dh = make_input(adata, n_nodes=k, mt_prior_strength=10., eps_prior_strength=2.,
                                    delta_prior_strength=0.09, step_size=0.3, debug=True, wis_sample_size=10)
        config.split = 'categorical'
        victree = VICTree(config, jq, data_handler=dh)
        victree.run(100)
        result = evaluate_victree_to_df(true_jq, victree, dataset_id=seed, tree_enumeration=True)
        print(result)

    def test_update_qc_qz(self):

        joint_q = self.generate_test_dataset_fixed_tree(step_size=0.3)
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
        config = Config(n_nodes=5, n_states=7, n_cells=200, chain_length=500,
                        wis_sample_size=50, step_size=.1,
                        debug=True)
        true_joint_q = generate_dataset_var_tree(config, eps_a=200., eps_b=20000., dir_alpha=3.,
                                                 lambda_prior=1000., alpha_prior=500., beta_prior=50.,
                                                 # cne_length_factor=50
                                                 )

        matplotlib.use('module://backend_interagg')
        utils.visualization_utils.plot_dataset(true_joint_q)['fig'].show()
        print("--- TRUE JOINT - NOT RE-LABELED ---")
        print(true_joint_q)
        qt = qT(config, sampling_method='mst')
        qmt = qMuTau(config, from_obs=(true_joint_q.obs, 20.))
        qeps = qEpsilonMulti(config, alpha_prior=20., beta_prior=2000.)
        joint_q = VarTreeJointDist(config, obs=true_joint_q.obs, qmt=qmt, qeps=qeps, qt=qt)
        joint_q.initialize()
        # joint_q.z.initialize(method='kmeans', data=true_joint_q.obs, skewness=2)
        joint_q.z.initialize(method='gmm', data=joint_q.obs)
        joint_q.mt.initialize(method='data-size', obs=true_joint_q.obs)
        joint_q.c.initialize(method='diploid')
        joint_q.eps.initialize(method='data', obs=true_joint_q.obs)
        for i in range(50):
            joint_q.update(i)

        gt_z = true_joint_q.z.true_params['z'].numpy()
        vi_z = joint_q.z.pi.numpy()
        best_mapp = best_mapping(gt_z, vi_z)
        print(best_mapp)
        true_tree = tree_utils.relabel_nodes(true_joint_q.t.true_params['tree'], best_mapp)
        print(f'\n\n--- TRUE TREE (after remap): {tree_to_newick(true_tree)}')

        print("\n\n--- VAR JOINT ---")
        print(joint_q)

        # compare with quality measures
        victree = VICTree(config, joint_q, joint_q.obs, draft=True)
        res_df = evaluate_victree_to_df(true_joint_q, victree, 0, tree_enumeration=True)
        print("--- EVALUATION --- ")
        for k, v in res_df.to_dict().items():
            print(f"{k}: {v}")

        sample_size = 100
        t_pmf: dict = joint_q.t.get_pmf_estimate(normalized=True, n=sample_size, desc_sorted=True)
        print(t_pmf)
        # NOTE: copy number is not very accurate and tree sampling is not exact, but still some
        #   of the true edges obtain high probability of being sampled.
        #   also, the weights don't explode to very large or very small values, causing the algorithm to crash

    def test_update_qz_qmt(self):
        joint_q = self.generate_test_dataset_fixed_tree()
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
        print(f"true mu: {joint_q.mt.true_params['mu']}")
        print(f"true tau: {joint_q.mt.true_params['tau']}")
        for i in range(n_iter):
            if i % 5 == 0:
                var_cellassignment = torch.max(qz.pi, dim=-1)[1]
                ari = adjusted_rand_score(joint_q.z.true_params['z'], var_cellassignment)
                print(f"[{i}]")
                print(f"- qz adjusted rand idx: {ari:.2f}")
                print(f"- qmt dist: {torch.pow(qmt.nu - joint_q.mt.true_params['mu'], 2).sum():.2f}")

            qmt.update(fix_qc, qz, obs)
            qz.update(qmt, fix_qc, fix_qpi, obs)
        print(f"results after {n_iter} iter")
        print(f"- var z: {torch.max(qz.pi, dim=-1)[1]}")
        print(f"- var mu: {qmt.nu}")
        print(f"- var tau: {qmt.exp_tau()}")

    def test_update_qc_qz_qmt(self):
        joint_q = self.generate_test_dataset_fixed_tree()
        cfg = joint_q.config
        obs = joint_q.obs
        fix_tree = joint_q.T
        fix_qpi = joint_q.pi
        fix_qeps = joint_q.eps

        qmt = qMuTau(cfg, from_obs=(obs, 1.))
        qz = qZ(cfg)
        qc = qC(cfg)
        qmt.initialize(method='data-size', obs=obs)
        almost_true_z_init = joint_q.z.exp_assignment() + .2
        almost_true_z_init /= almost_true_z_init.sum(dim=1, keepdim=True)
        qz.initialize(z_init='fixed', pi_init=almost_true_z_init)
        # qz.initialize(method='kmeans', obs=obs)
        qc.initialize()

        trees = [fix_tree] * cfg.wis_sample_size
        wis_weights = [1 / cfg.wis_sample_size] * cfg.wis_sample_size

        # change step_size
        cfg.step_size = .3

        for i in range(20):
            qz.update(qmt, qc, fix_qpi, obs)
            qmt.update(qc, qz, obs)
            qc.update(obs, fix_qeps, qz, qmt, trees=trees, tree_weights=wis_weights)

        ari = adjusted_rand_score(joint_q.z.true_params['z'], qz.best_assignment())
        self.assertGreater(ari, .9)
        cn_mad = torch.mean(torch.abs(joint_q.c.true_params['c'] - qc.get_viterbi()).float())
        self.assertLess(cn_mad, 0.01)

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
        #utils.visualization_utils.visualize_copy_number_profiles(true_cn_profile)
        print(f"true z: {true_z}")
        print(f"[init] var z: {qz.exp_assignment()}")
        for i in range(30):
            joint_q.update(i)
            if i % 5 == 0:
                #utils.visualization_utils.visualize_copy_number_profiles(
                #    torch.argmax(joint_q.c.single_filtering_probs, dim=-1))
                print(f"[{i}] var z: {qz.exp_assignment()}")

                trees, weights = joint_q.t.get_trees_sample(sample_size=10)
                print(f"[{i}] elbo: {joint_q.compute_elbo(trees, weights)}")

        # same but with z at current pos

    def test_update_qz_qeps_qt(self):
        # fix qc, qpi and qmt
        config = Config(n_nodes=7, n_states=7, n_cells=200, chain_length=500,
                        wis_sample_size=20, step_size=.3,
                        debug=True)
        joint_q = generate_dataset_var_tree(config)
        qt = qT(config, norm_method='stochastic', sampling_method='rand-mst')
        qt.initialize()
        qz = qZ(config)
        qz.initialize()
        qeps = qEpsilonMulti(config, alpha_prior=50., beta_prior=4000.)
        qeps.initialize(method='data', obs=joint_q.obs)
        for i in range(40):
            qt.update(joint_q.c, qeps)
            trees, log_w = qt.get_trees_sample()
            qeps.update(trees, log_w, joint_q.c)
            qz.update(joint_q.mt, joint_q.c, joint_q.pi, joint_q.obs)

        print("\n\n --- TRUE DIST ---\n")
        print(joint_q)

        print("\n\n --- VAR DIST ---\n")
        print(qt)
        print(qeps)
        print(qz)
        # assert that the first tree among the sampled ones is the true one
        self.assertEqual(list(qt.get_pmf_estimate(desc_sorted=True).keys())[0],
                         tree_to_newick(joint_q.t.true_params['tree']))

    def test_update_qz_qeps_qt_qc(self):
        # fix qmt
        config = Config(n_nodes=7, n_states=7, n_cells=200, chain_length=500,
                        wis_sample_size=20, step_size=.3,
                        debug=True)
        joint_q = generate_dataset_var_tree(config, eps_a=50., eps_b=4000.)
        qt = qT(config, norm_method='stochastic')
        qt.initialize()
        qz = qZ(config)
        qz.initialize()
        qeps = qEpsilonMulti(config, alpha_prior=50., beta_prior=4000.)
        qeps.initialize(method='data', obs=joint_q.obs)
        qc = qC(config)
        qc.initialize(method='diploid')

        for i in range(50):
            qt.update(qc, qeps)
            trees, weights = qt.get_trees_sample(alg='rand-mst')
            qeps.update(trees, weights, qc)
            qz.update(joint_q.mt, qc, joint_q.pi, joint_q.obs)
            qc.update(joint_q.obs, qeps, qz, joint_q.mt, trees, weights)
        print(qt)

        for i in range(10):
            # refine
            mst_list, _ = qt.get_trees_sample(alg='mst', sample_size=1)
            unit_weights = [1.]
            qeps.update(mst_list, unit_weights, joint_q.c)
            qz.update(joint_q.mt, joint_q.c, joint_q.pi, joint_q.obs)
            qc.update(joint_q.obs, qeps, qz, joint_q.mt, mst_list, unit_weights)

        print("\n\n --- TRUE DIST ---\n")
        print(joint_q)
        mapp = best_mapping(joint_q.z.true_params['z'].numpy(), qz.pi.numpy())
        remapped_true_tree = nx.relabel_nodes(joint_q.t.true_params['tree'],
                                              {i: mapp[i] for i in range(config.n_nodes)})
        print(f" REMAPPED TRUE TREE: {tree_to_newick(remapped_true_tree)}")
        print(f"v-measure: {v_measure_score(joint_q.z.true_params['z'], qz.best_assignment())}")

        print("\n\n --- VAR DIST ---\n")
        print(qt)
        print(qeps)
        print(qz)
        print(qc)

if __name__ == '__main__':
    unittest.main()
