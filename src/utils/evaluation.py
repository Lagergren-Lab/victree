import itertools
import traceback

import anndata
import networkx as nx
import pandas as pd

from scipy.stats import poisson
import numpy as np
import math

from sklearn.metrics import adjusted_rand_score, v_measure_score

from simul import generate_dataset_var_tree
from utils import tree_utils
from utils.config import set_seed, Config
from utils.tree_utils import tree_to_newick, reorder_newick
from variational_distributions.joint_dists import JointDist


# FROM CopyMix
# TODO: DIC (see formula on wiki, this function implements that)
#   elbow method https://en.wikipedia.org/wiki/Deviance_information_criterion
#   to be preferred wrt log-likelihood (maybe)
def get_dic(self, clusters, cells):
    # calculate expected_hidden
    expected_hidden = np.zeros((self.K, self.J, self.M))
    sum_of_expected_hidden_two = np.zeros((self.K, self.J, self.J))
    for k in range(self.K):
        sum_of_expected_hidden_two[k] = clusters[k].sum_of_expectation_two()
        expected_hidden[k] = clusters[k].get_expectation()

    def handle_inf(x):
        import sys
        if np.isinf(x):
            return sys.maxsize
        else:
            return x

    # term_1 : -4 E[ log[p(Y | Z, C, Ψ)] ] w.r.t. final posterior values
    # term_2 : 2 log[p(Y|Z,C,Ψ)] where Z, C and Ψ are the modes (maximizing the posteriors)
    res = 0
    for n in range(self.N):
        for k in range(self.K):
            for j in range(self.J):
                res += np.sum(self.pi[n, k] * expected_hidden[k, j, :] *
                              calculate_expectation_of_D(j, self.epsilon_r[n], self.epsilon_s[n], cells[n]))
    term_1 = - res

    res = 0
    for n in range(self.N):
        for m in range(self.M):
            k = int(np.argmax(self.pi[n, :]))
            j = int(np.argmax(expected_hidden[k, :, m]))
            theta = self.epsilon_s[n] / self.epsilon_r[n]
            state = add_noise(j)
            D = handle_inf(math.log(poisson.pmf(cells[n, m], theta * state) + .0000001))
            res += D
    term_2 = 2 * res

    return term_1, 4 * term_1 + term_2

def calculate_expectation_of_D(j, epsilon_r, epsilon_s, cell):
    pass

def add_noise(j):
    if j == 0:
        j = 0.001
    return j


def best_mapping(gt_z: np.ndarray, vi_z: np.ndarray, with_score=False):
    """
    Returns best permutation to go from gt_lab -> vi_lab
    Use it to change ground truth labels
    Parameters
    ----------
    gt_z: np.ndarray w/ shape (N,)
    vi_z: np.ndarray w/ shape (N, K)
    """
    K = vi_z.shape[1]
    assert np.unique(gt_z).size == K, f"{np.unique(gt_z).size} != {K}"
    perms = [list((0,) + p) for p in itertools.permutations(range(1, K))]
    if len(perms) > 10e7:
        print(f"warn: num of permutations is {len(perms)}")
    # get one-hot ground truth z
    one_hot_gt = np.eye(K)[gt_z]
    scores = []
    for p in perms:
        score = np.sum(vi_z * one_hot_gt[:, p])
        scores.append(score)

    best_perm_idx = np.argmax(scores)
    if with_score:
        return perms[best_perm_idx], scores[best_perm_idx]
    else:
        return perms[best_perm_idx]


def evaluate_victree_to_df(true_joint, victree, dataset_id, df=None, tree_enumeration=False):
    """
    Appends evaluation info
    Parameters
    ----------
    true_joint
    victree
    dataset_id
    df: pandas.DataFrame, existing dataframe on which to append scores row

    Returns
    -------

    """
    out_data = {}
    out_data['dataset-id'] = dataset_id
    out_data['K'] = true_joint.config.n_nodes
    out_data['vK'] = victree.config.n_nodes
    out_data['M'] = true_joint.config.chain_length
    out_data['N'] = true_joint.config.n_cells
    out_data['true-ll'] = true_joint.total_log_likelihood
    out_data['vi-ll'] = victree.q.total_log_likelihood
    out_data['vi-diff'] = out_data['true-ll'] - out_data['vi-ll']
    out_data['elbo'] = victree.elbo
    out_data['iters'] = victree.it_counter
    out_data['time'] = victree.exec_time_

    # clustering eval
    true_lab = true_joint.z.true_params['z']
    vi_lab = victree.q.z.best_assignment()
    out_data['ari'] = adjusted_rand_score(true_lab, vi_lab)
    out_data['v-meas'] = v_measure_score(true_lab, vi_lab)

    # copy number calling eval
    true_c = true_joint.c.true_params['c'][true_lab].numpy()
    pred_c = victree.q.c.get_viterbi()[vi_lab].numpy()
    cn_mad = np.abs(pred_c - true_c).mean()
    out_data['cn-mad'] = cn_mad

    # tree eval
    if victree.config.n_nodes == true_joint.config.n_nodes:
        best_map = best_mapping(true_lab, victree.q.z.pi.numpy())
        true_tree = tree_utils.relabel_nodes(true_joint.t.true_params['tree'], best_map)
        mst = nx.maximum_spanning_arborescence(victree.q.t.weighted_graph)
        intersect_edges = nx.intersection(true_tree, mst).edges
        out_data['edge-sensitivity'] = len(intersect_edges) / len(mst.edges)
        out_data['edge-precision'] = len(intersect_edges) / len(true_tree.edges)

        qt_pmf = victree.q.t.get_pmf_estimate(True, n=50)
        true_tree_newick = tree_to_newick(true_tree)
        mst_newick = tree_to_newick(mst)
        out_data['qt-true'] = qt_pmf[true_tree_newick].item() if true_tree_newick in qt_pmf.keys() else 0.
        out_data['qt-mst'] = qt_pmf[mst_newick].item() if mst_newick in qt_pmf.keys() else 0.
        pmf_arr = np.array(list(qt_pmf.values()))
        out_data['pt-true'] = np.nan
        out_data['pt-mst'] = np.nan
        if tree_enumeration:
            try:
                pt = victree.q.t.enumerate_trees()
                pt_dict = {tree_to_newick(nwk): math.exp(logp) for nwk, logp in zip(pt[0], pt[1].tolist())}
                out_data['pt-true'] = pt_dict[true_tree_newick]
                out_data['pt-mst'] = pt_dict[mst_newick]
            except BaseException:
                print(traceback.format_exc())

        # normalized entropy
        out_data['qt-entropy'] = - np.sum(pmf_arr * np.log(pmf_arr)) / np.log(pmf_arr.size)

    if df is None:
        df = pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([out_data])], ignore_index=True)
    return df


def check_clone_uniqueness(cn_mat):
    norm_mat = np.linalg.norm(cn_mat[:, np.newaxis] - cn_mat, axis=-1)
    non_diagonal = ~np.eye(cn_mat.shape[0], dtype=bool)
    return np.all(norm_mat[non_diagonal] > 0)


def sample_dataset_generation(K=4, M=500, N=200, seed=0, mut_rate: int = 1) -> (JointDist, anndata.AnnData):
    """
    Generate simulated data with various sizes and complexity (in terms of mutations over the whole genome)
    Parameters
    ----------
    K M N: int, respectively number of nodes, number of bins and number of cells
    seed: randomizer
    mut_rate: int, from 1 (low) to higher values
    """
    set_seed(seed)
    # set variance depending on size of dataset
    # eps is going to have 5 asymmetric mutations over the whole sequence
    # but in shorter sequences, variance must be lower
    eps_a = 5 * 100 * 500 / M * mut_rate
    eps_b = eps_a / mut_rate / 3 * M

    dir_alpha = 2 * 1000 / N

    cne_length_factor = 0
    # cne_length_factor = M / 50

    # simulate data such that every clone is different
    is_unique = False
    while not is_unique:
        joint_q_true, adata = generate_dataset_var_tree(config=Config(
            n_nodes=K, n_cells=N, chain_length=M, eps0=.1
        ), ret_anndata=True, chrom='real', dir_alpha=3., eps_a=eps_a, eps_b=eps_b,
            nu_prior=1., lambda_prior=1000., alpha_prior=500., beta_prior=50.,
            cne_length_factor=cne_length_factor)
        is_unique = check_clone_uniqueness(joint_q_true.c.true_params['c'])

    return joint_q_true, adata
