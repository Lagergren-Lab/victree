import unittest

import numpy as np
from sklearn.metrics import v_measure_score
import matplotlib.pyplot as plt

from inference.victree import make_input, VICTree
from simul import generate_dataset_var_tree
from tests.utils_testing import print_logs
from utils.config import Config, set_seed
from utils.evaluation import best_mapping, evaluate_victree_to_df, sample_dataset_generation
from utils.visualization_utils import plot_cn_matrix


class VICTreeTestCase(unittest.TestCase):

    def setUp(self) -> None:
        set_seed(0)
        print_logs("DEBUG")

    def test_fixed_tree_run(self):
        # simulate data
        joint_q_true, adata = generate_dataset_var_tree(config=Config(
            n_nodes=4, n_cells=100, chain_length=300, wis_sample_size=50,
        ), ret_anndata=True, chrom=3, dir_alpha=10., eps_a=25., eps_b=10000.)
        print(f"true dist log-likelihood {joint_q_true.total_log_likelihood}")
        # make default input
        config, q, dh = make_input(adata, fix_tree=joint_q_true.t.true_params['tree'], debug=True,
                                   step_size=.4, mt_prior_strength=5., delta_prior_strength=2.,
                                   eps_prior_strength=2., c_init='diploid')

        # check victree convergence
        init_elbo = q.elbo
        print(f"init elbo: {init_elbo}")
        victree = VICTree(config, q, data_handler=dh, elbo_rtol=1e-4)
        victree.run(n_iter=60)
        self.assertGreater(victree.elbo, init_elbo, f"elbo diff: {victree.elbo - init_elbo}")

        true_lab = joint_q_true.z.true_params['z']
        pred_lab = q.z.pi.argmax(dim=1)
        v_score = v_measure_score(true_lab, pred_lab)
        # print(f"v-measure score: {v_score:.3f}")
        self.assertGreater(v_score, 0.8)

        best_map = best_mapping(true_lab, q.z.pi.numpy())
        # print(f"best map: {best_map}")
        true_c = joint_q_true.c.true_params['c'][best_map].numpy()
        pred_c = q.c.get_viterbi().numpy()
        # print(f"true c: {true_c}")
        # print(f"pred c: {pred_c}")
        cn_mad = np.abs(pred_c - true_c).mean()
        fig, axs = plt.subplots(2, 2)
        plot_cn_matrix(true_c, np.array(best_map)[true_lab], axs=axs[:, 0])
        plot_cn_matrix(pred_c, pred_lab, axs=axs[:, 1])
        axs[0, 0].set_title('True CN')
        axs[0, 1].set_title('VI CN')
        fig.savefig("./../test_output/test_fixed_tree_run_cn.png")
        # print(f"cn matrix mean abs deviation: {cn_mad:.3f}")
        self.assertLess(cn_mad, 0.1)

    def test_victree_update_less_t(self):
        """
        Uses the update schema by which a tree sample is taken only
        every 5 iterations
        """
        seed = 101
        K = 4
        print("sampling dataset")
        joint_q_true, adata = sample_dataset_generation(K=K, seed=seed)
        config, jq, dh = make_input(adata, n_nodes=K, mt_prior_strength=5.,
                                    delta_prior_strength=0.01, eps_prior_strength=0.01,
                                    step_size=0.4, split='categorical')
        # config.decay = 0.001
        config.update_type = 'less-t'
        victree = VICTree(config, jq, data_handler=dh, draft=True)
        victree.run(100)

        eval_df = evaluate_victree_to_df(joint_q_true, victree, 0)
        eval_dict = eval_df.iloc[0].to_dict()
        print(eval_dict)
        self.assertGreater(eval_dict['ari'], 0.7)
        self.assertGreater(eval_dict['v-meas'], 0.7)
        self.assertLess(eval_dict['cn-mad'], 0.1)
        self.assertGreater(eval_dict['q-true'], 0.)
        self.assertGreater(eval_dict['q-mst'], 0.)

    def test_victree_ss_decay(self):
        """
        Uses time-based decay for step-size scheduling.
        """
        seed = 101
        K = 4
        print("sampling dataset")
        joint_q_true, adata = sample_dataset_generation(K=K, seed=seed)
        config, jq, dh = make_input(adata, n_nodes=K, mt_prior_strength=5.,
                                    delta_prior_strength=0.01, eps_prior_strength=0.01,
                                    step_size=1., split='categorical')
        config.decay = 0.001
        # config.update_type = 'less-t'
        victree = VICTree(config, jq, data_handler=dh, draft=True)
        victree.run(100)

        eval_df = evaluate_victree_to_df(joint_q_true, victree, 0)
        eval_dict = eval_df.iloc[0].to_dict()
        print(eval_dict)
        self.assertGreater(eval_dict['ari'], 0.7)
        self.assertGreater(eval_dict['v-meas'], 0.7)
        self.assertLess(eval_dict['cn-mad'], 0.1)
        self.assertGreater(eval_dict['q-true'], 0.)
        self.assertGreater(eval_dict['q-mst'], 0.)


if __name__ == '__main__':
    unittest.main()
