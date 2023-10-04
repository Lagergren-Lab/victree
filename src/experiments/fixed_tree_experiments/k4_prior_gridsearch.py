import calendar
import itertools
import math
import os.path
import time

import anndata
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, v_measure_score

from utils.config import set_seed, Config
from simul import generate_dataset_var_tree
from inference.victree import VICTree, make_input
from utils.evaluation import best_mapping
from utils.visualization_utils import plot_cn_matrix
from variational_distributions.joint_dists import JointDist


def sample_dataset_generation(seed=0) -> (JointDist, anndata.AnnData):
    set_seed(seed)

    # simulate data
    joint_q_true, adata = generate_dataset_var_tree(config=Config(
        n_nodes=4, n_cells=100, chain_length=300, wis_sample_size=50,
    ), ret_anndata=True, chrom=3, dir_alpha=10., eps_a=25., eps_b=10000.)

    return joint_q_true, adata


if __name__ == "__main__":
    out_path = f"./{calendar.timegm(time.gmtime())}_k4_prior_gridsearch.csv"
    n_datasets = 10
    # n_datasets = 2
    max_iter = 100
    # max_iter = 3
    prior_strength = [0.05, 1., 5.]
    # prior_strength = [1., 5.]
    with_sieving = False
    param_search = {
        'mt_prior': prior_strength,
        'eps_prior': prior_strength,
        'delta_prior': prior_strength,
        'step_size': [0.3, 0.6, 0.1]
    }
    grid_search_list = list(itertools.product(*param_search.values()))
    print(f"executing {len(grid_search_list)} x {n_datasets} runs (configs x n_datasets)")
    datasets_path = "./sample_datasets"
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
    for d in range(n_datasets):
        joint_q_true, adata = sample_dataset_generation(seed=d)
        pl = plot_cn_matrix(joint_q_true.c.get_viterbi(), joint_q_true.z.best_assignment())
        pl['fig'].savefig(os.path.join(datasets_path, f"{d}.png"))

        for i, params in enumerate(grid_search_list):
            print(f"[{i}/{len(grid_search_list)}] dat {d} with {param_search.keys()} = {params}")
            mt_ps, eps_ps, delta_ps, step_size = params
            config, q, dh = make_input(adata, 'copy', fix_tree=joint_q_true.t.true_params['tree'],
                                       mt_prior_strength=mt_ps, eps_prior_strength=eps_ps,
                                       delta_prior_strength=delta_ps,
                                       step_size=step_size,
                                       sieving=(3, math.floor(3 / step_size)) if with_sieving else (1, 1))

            victree = VICTree(config, q, data_handler=dh)
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


