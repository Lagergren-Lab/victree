#!/usr/bin/env python3
import os.path
import re
import sys

from inference.victree import make_input, VICTree
from utils.evaluation import sample_dataset_generation, evaluate_victree_to_df


def run_dataset(K, M, N, mr, seed):
    out_path = f"./dat{seed}_K{K}M{M}N{N}mr{mr}"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    print(f"running dat: K {K} M {M} N {N} mut rate {mr}, dataset {seed}")
    jq_true, ad = sample_dataset_generation(K, M, N, seed, mut_rate=mr)
    # execute with K = trueK -1 and trueK + 1
    n_nodes = K
    n_bins = jq_true.config.chain_length
    n_cells = jq_true.config.n_cells
    config, jq, dh = make_input(ad, n_nodes=n_nodes, mt_prior=(1., 20. * n_bins, 500., 50),
                                eps_prior=(5., 1. * n_bins), delta_prior=3.,
                                z_init='gmm', c_init='diploid', mt_init='data-size',
                                kmeans_skewness=3,
                                step_size=0.3,
                                # sieving=(3, 3),
                                split='ELBO',
                                debug=True)
    victree = VICTree(config, jq, data_handler=dh, draft=True, elbo_rtol=1e-4)
    victree.run(100)

    # save results
    results_df = evaluate_victree_to_df(jq_true, victree, dataset_id=seed,
                                        tree_enumeration=n_nodes < 7)
    results_df['mr'] = mr
    out_csv = os.path.join(out_path, f"score_mr.csv")
    print(f"results saved in: {out_csv}")
    results_df.to_csv(out_csv, index=False)


if __name__ == '__main__':
    # read dataset path
    dat_path = sys.argv[1]
    params_re = re.compile(r'^.*/K(?P<K>\d+)M(?P<M>\d+)N(?P<N>\d+)mr(?P<mr>\d+)/(?P<seed>\d+)\.png$')
    re_match = params_re.match(dat_path)
    run_dataset(int(re_match.group('K')), int(re_match.group('M')), int(re_match.group('N')),
                int(re_match.group('mr')), int(re_match.group('seed')))
