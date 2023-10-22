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


def complete_database(size: int, K: list[int], M: list[int], N: list[int]):
    datasets_path = "../sample_datasets/"
    for k, m, n in zip(K, M, N):
        k_dat_path = os.path.join(datasets_path, f"K{k}M{m}N{n}")
        if not os.path.exists(k_dat_path):
            os.mkdir(k_dat_path)
        # extract the name of <seeds>.png files
        seeds = [int(os.path.splitext(fn)[0]) for fn in os.listdir(k_dat_path)]
        ans = ''
        while len(seeds) < size:
            rnd_seed = random.randint(0, 10000)
            if rnd_seed in seeds:
                continue
            set_seed(rnd_seed)
            joint_q_true, adata = sample_dataset_generation(K=k, M=m, N=n, seed=rnd_seed)
            pl = plot_dataset(joint_q_true)
            pl['fig'].show()

            while ans not in ['y', 'n', 'a']:
                ans = input(f"save dataset #{rnd_seed}? ({len(seeds)}/{n_datasets}) y (yes) / n (no) / a (all): ")
            if ans in ['y', 'a']:
                pl['fig'].savefig(os.path.join(k_dat_path, f"{rnd_seed}.png"))
                seeds.append(rnd_seed)
                # complete without further asking until next setting
                # by fixing ans to 'a'
            if ans != 'a':
                # reset in order to ask again
                ans = ''


def sample_dataset_generator(size: int, K: list[int], M: list[int], N: list[int]):
    datasets_path = "../sample_datasets/"
    for k, m, n in zip(K, M, N):
        # extract the name of <seeds>.png files
        seeds = [int(os.path.splitext(fn)[0]) for fn in os.listdir(os.path.join(datasets_path, f"K{k}M{m}N{n}"))]
        if len(seeds) < size:
            print("ERROR: not enough datasets, run with v flag")
            exit(1)
        for s in random.sample(seeds, size):
            yield s, sample_dataset_generation(K=k, M=m, N=n, seed=s)


if __name__ == '__main__':
    n_datasets = 20
    # n_datasets = 1
    K_list = [
        # 6,
        9,
        # 12
    ]
    M_list = [
        # 1000,
        3000,
        # 5000
    ]
    N_list = [
        # 300,
        600,
        # 1000
    ]
    # K_list = [4, 6]
    sys.argv.append('v')

    if len(sys.argv) > 1:
        if sys.argv[1] == 'v':
            matplotlib.use('module://backend_interagg')
            complete_database(n_datasets, K_list, M_list, N_list)
            print("Database complete")
            exit(0)

    # init csv out file
    results_df = None
    out_csv = f"./{calendar.timegm(time.gmtime())}_performance.csv"
    # out_csv = f"./tmp_{calendar.timegm(time.gmtime())}_performance.csv"
    dat_counter = 0
    n_nodes, n_bins, n_cells = list(zip(K_list, M_list, N_list))[0]
    # load datasets
    for dataset_id, (jq_true, ad) in sample_dataset_generator(n_datasets, K=K_list, M=M_list, N=N_list):
        # track progress on std out, reset counter when type of dataset changes
        if (jq_true.config.n_nodes, jq_true.config.chain_length, jq_true.config.n_cells) != (n_nodes, n_bins, n_cells):
            dat_counter = 0
        n_nodes = jq_true.config.n_nodes
        n_bins = jq_true.config.chain_length
        n_cells = jq_true.config.n_cells
        print(f"K {n_nodes} M {n_bins} N {n_cells} - {dat_counter} / {n_datasets}")

        # run victree
        config, jq, dh = make_input(ad, n_nodes=n_nodes, mt_prior=(1., 20. * n_bins, 500., 50),
                                    eps_prior=(5., 1. * n_bins), delta_prior=3.,
                                    z_init='gmm', c_init='diploid', mt_init='data-size',
                                    kmeans_skewness=3,
                                    step_size=0.3, wis_sample_size=10,
                                    # sieving=(3, 3),
                                    split='ELBO',
                                    debug=True)
        victree = VICTree(config, jq, data_handler=dh, draft=True, elbo_rtol=1e-4)
        victree.run(50)

        # save results
        results_df = evaluate_victree_to_df(jq_true, victree, dataset_id=dataset_id, df=results_df,
                                            tree_enumeration=n_nodes < 7)
        results_df.to_csv(out_csv, index=False)
        dat_counter += 1
