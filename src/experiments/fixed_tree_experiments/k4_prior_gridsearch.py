import calendar
import itertools
import math
import os.path
import random
import time

import anndata
import matplotlib
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, v_measure_score

from utils.config import set_seed, Config
from simul import generate_dataset_var_tree
from inference.victree import VICTree, make_input
from utils.evaluation import best_mapping
from utils.visualization_utils import plot_dataset
from variational_distributions.joint_dists import JointDist


def check_clone_uniqueness(cn_mat):
    norm_mat = np.linalg.norm(cn_mat[:, np.newaxis] - cn_mat, axis=-1)
    non_diagonal = ~np.eye(cn_mat.shape[0], dtype=bool)
    return np.all(norm_mat[non_diagonal] > 0)


def sample_dataset_generation(K=4, seed=0) -> (JointDist, anndata.AnnData):
    set_seed(seed)

    # simulate data such that every clone is different
    is_unique = False
    while not is_unique:
        joint_q_true, adata = generate_dataset_var_tree(config=Config(
            n_nodes=K, n_cells=200, chain_length=500, wis_sample_size=50,
        ), ret_anndata=True, chrom=3, dir_alpha=3., eps_a=50., eps_b=10000.,
            cne_length_factor=0)
        is_unique = check_clone_uniqueness(joint_q_true.c.true_params['c'])

    return joint_q_true, adata


if __name__ == "__main__":

    matplotlib.use('module://backend_interagg')

    out_path = f"./{calendar.timegm(time.gmtime())}_k4_prior_gridsearch.csv"
    n_datasets = 15
    # n_datasets = 2
    max_iter = 200
    # max_iter = 2
    elbo_rtol = 5e-5
    with_sieving = False
    param_search = {
        'mt_ps': [5.],
        'eps_ps': [0.5, 0.05],
        'delta_ps': [0.5, 0.05],
        'step_size': [0.3, 0.1]
    }
    grid_search_list = list(itertools.product(*param_search.values()))
    print(f"executing {len(grid_search_list)} x {n_datasets} runs (configs x n_datasets)")
    datasets_path = "../sample_datasets"
    if not os.path.exists(datasets_path):
        os.mkdir(datasets_path)

    out_data = {
        "true_ll": [],
        "dat": [],
        "mt_ps": [],
        "eps_ps": [],
        "delta_ps": [],
        "final_ll": [],
        "iters": [],
        "elbo": [],
        "cn-mad": [],
        "ari": [],
        "v-meas": [],
        "step_size": []
    }
    # init to default datasets seeds if already available
    dataset_seeds = [11, 140, 168, 224, 262, 273, 312,
                342, 462, 551, 572, 603, 742, 836,
                848, 887, 938]

    datasets = []
    # if n_datasets is less, just sample those that are asked
    if len(dataset_seeds) > n_datasets:
        dataset_seeds = [i for i in random.sample(dataset_seeds, n_datasets)]

    # generate those datasets
    for ds in dataset_seeds:
        set_seed(ds)
        datasets.append((ds, sample_dataset_generation(seed=ds)))

    # otherwise, fill up datasets with new ones (input required in initial phase)
    while len(datasets) < n_datasets:
        rnd_seed = random.randint(0, 1000)
        set_seed(rnd_seed)
        joint_q_true, adata = sample_dataset_generation(seed=rnd_seed)
        pl = plot_dataset(joint_q_true)
        pl['fig'].show()

        ans = ''
        while ans not in ['y', 'n']:
            ans = input(f"save dataset #{rnd_seed}? ({len(datasets)}/{n_datasets}) y/n: ")
        if ans == 'y':
            pl['fig'].savefig(os.path.join(datasets_path, f"{rnd_seed}.png"))
            datasets.append((rnd_seed, (joint_q_true, adata)))

    # run grid search over multiple datasets
    for d, (joint_q_true, adata) in datasets:

        for i, params in enumerate(grid_search_list):
            print(f"[{i}/{len(grid_search_list)}] dat {d} with {param_search.keys()} = {params}")
            mt_ps, eps_ps, delta_ps, step_size = params
            config, q, dh = make_input(adata, 'copy', fix_tree=joint_q_true.t.true_params['tree'],
                                       mt_prior_strength=mt_ps, eps_prior_strength=eps_ps,
                                       delta_prior_strength=delta_ps,
                                       step_size=step_size,
                                       z_init='random', c_init='diploid',
                                       sieving=(3, math.floor(3 / step_size)) if with_sieving else (1, 1))

            victree = VICTree(config, q, data_handler=dh, elbo_rtol=elbo_rtol)
            victree.run(n_iter=max_iter)

            # save run
            out_data['true_ll'].append(joint_q_true.total_log_likelihood)
            out_data['dat'].append(d)
            out_data['step_size'].append(step_size)
            out_data['mt_ps'].append(mt_ps)
            out_data['eps_ps'].append(eps_ps)
            out_data['delta_ps'].append(delta_ps)

            out_data['final_ll'].append(victree.q.total_log_likelihood)
            out_data['iters'].append(victree.it_counter)
            out_data['elbo'].append(victree.elbo)

            true_lab = joint_q_true.z.true_params['z']
            out_data['ari'].append(adjusted_rand_score(true_lab, q.z.best_assignment()))
            out_data['v-meas'].append(v_measure_score(true_lab, q.z.best_assignment()))
            best_map = best_mapping(true_lab, q.z.pi.numpy())
            true_c = joint_q_true.c.true_params['c'][best_map].numpy()
            pred_c = q.c.get_viterbi().numpy()
            cn_mad = np.abs(pred_c - true_c).mean()
            out_data['cn-mad'].append(cn_mad)

            print("\tEND: {", *[f"{k}: {v[-1]}" for k, v in out_data.items()], "}")

    df = pd.DataFrame(out_data)
    df.to_csv(out_path, index=False)

    # printout brief summary
    # find top three config based on likelihood difference

    df['ll_rel_diff'] = - np.abs(df['true_ll'] - df['final_ll']) / df['true_ll'] * 100
    print(f"See average of log-lik relative difference in % over {n_datasets}"
          f" different datasets")
    print(df.groupby(list(param_search.keys()))['ll_rel_diff'].mean())




