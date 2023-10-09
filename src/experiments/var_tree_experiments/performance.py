"""
Experiments for evaluating overall performance of VICTree with
qT variational distribution.
We evaluate based on comparison against ground-truth from simulated data
- clustering accuracy (ARI and V-measure scores)
- copy number calling (MAD)
- tree reconstruction (edge sensitivity/precision and q(T_true), entropy of q)
- overall quality (ELBO and log-likelihood)
"""
import calendar
import random
import sys
import os
import time

import matplotlib

from inference.victree import make_input, VICTree
from utils.config import set_seed
from utils.evaluation import evaluate_victree_to_df, sample_dataset_generation
from utils.visualization_utils import plot_dataset


def complete_database(size: int, K: list[int]):
    datasets_path = "../sample_datasets/"
    for k in K:
        k_dat_path = os.path.join(datasets_path, f"K{k}")
        if not os.path.exists(k_dat_path):
            os.mkdir(k_dat_path)
        # extract the name of <seeds>.png files
        seeds = [int(os.path.splitext(fn)[0]) for fn in os.listdir(k_dat_path)]
        while len(seeds) < size:
            rnd_seed = random.randint(0, 10000)
            if rnd_seed in seeds:
                continue
            set_seed(rnd_seed)
            joint_q_true, adata = sample_dataset_generation(K=k, seed=rnd_seed)
            pl = plot_dataset(joint_q_true)
            pl['fig'].show()

            ans = ''
            while ans not in ['y', 'n']:
                ans = input(f"save dataset #{rnd_seed}? ({len(seeds)}/{n_datasets}) y/n: ")
            if ans == 'y':
                pl['fig'].savefig(os.path.join(k_dat_path, f"{rnd_seed}.png"))
                seeds.append(rnd_seed)


def sample_dataset_generator(size: int, K: list[int]):
    datasets_path = "../sample_datasets/"
    for k in K:
        # extract the name of <seeds>.png files
        seeds = [int(os.path.splitext(fn)[0]) for fn in os.listdir(os.path.join(datasets_path, f"K{k}"))]
        if len(seeds) < size:
            print("ERROR: not enough datasets, run with v flag")
            exit(1)
        for s in random.sample(seeds, size):
            yield s, sample_dataset_generation(K=k, seed=s)


if __name__ == '__main__':
    n_datasets = 20
    # n_datasets = 2
    K_list = [4, 6, 8]
    # K_list = [4, 6]
    # sys.argv.append('v')

    if len(sys.argv) > 1:
        if sys.argv[1] == 'v':
            matplotlib.use('module://backend_interagg')
            complete_database(n_datasets, K_list)

    # init csv out file
    results_df = None
    out_csv = f"./{calendar.timegm(time.gmtime())}_performance.csv"
    # out_csv = f"./tmp_{calendar.timegm(time.gmtime())}_performance.csv"
    dat_counter = 0
    n_nodes = K_list[0]
    # load datasets
    for dataset_id, (jq_true, ad) in sample_dataset_generator(n_datasets, K=K_list):
        # track progress on std out
        if jq_true.config.n_nodes != n_nodes:
            dat_counter = 0
        n_nodes = jq_true.config.n_nodes
        print(f"K {n_nodes} - {dat_counter} / {n_datasets}")

        # run victree
        config, jq, dh = make_input(ad, n_nodes=n_nodes, mt_prior_strength=10.,
                                    eps_prior_strength=10., delta_prior_strength=0.08,
                                    z_init='kmeans', c_init='diploid', mt_init='data-size',
                                    step_size=0.3, wis_sample_size=n_nodes,
                                    # sieving=(3, 3),
                                    split='categorical',
                                    debug=True)
        victree = VICTree(config, jq, data_handler=dh, draft=True)
        victree.run(100)

        # save results
        results_df = evaluate_victree_to_df(jq_true, victree, dataset_id=dataset_id, df=results_df)
        results_df.to_csv(out_csv, index=False)
        dat_counter += 1
