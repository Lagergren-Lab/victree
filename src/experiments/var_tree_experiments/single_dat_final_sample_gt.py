#!/usr/bin/env python3

"""
This experiments aims at finding the right temperature for g(T) when sampling
at the end of inference in order to get a better (less peaked) view of the q(T) distribution

It executes inference on a dataset, then computes a sample with different temperature values.
Computes inference scores/output measures and saves it on a CSV file.

USAGE:

e.g.
`./single_dat_final_sample_gt.py ./K9M3000N600/123.png 0.3,1.2,4.0`
where the list of floats determine the relative temperature of G wrt to qT in
the sample for scoring, not during inference.
"""

import os.path
import re
import sys

from inference.victree import make_input, VICTree
from utils.evaluation import sample_dataset_generation, evaluate_victree_to_df


def gen_data(K, M, N, seed):
    jq_true, ad = sample_dataset_generation(K, M, N, seed)
    return jq_true, ad


def run_dataset(jq_true, ad, extend_qt_temp=1., gt_temp_mult=1.):
    n_nodes = jq_true.config.n_nodes
    n_bins = jq_true.config.chain_length
    config, jq, dh = make_input(ad, n_nodes=n_nodes, mt_prior=(1., 20. * n_bins, 500., 50),
                                eps_prior=(5., 1. * n_bins), delta_prior=3.,
                                z_init='gmm', c_init='diploid', mt_init='data-size',
                                step_size=0.3,
                                split='ELBO',
                                debug=True)
    config.gT_temp = gt_temp_mult * config.qT_temp
    config.temp_extend = extend_qt_temp
    victree = VICTree(config, jq, data_handler=dh, draft=True, elbo_rtol=1e-4)
    victree.run(100, final_step=True)
    return victree


def compute_scores(jq_true, victree, seed, final_temps):
    n_nodes = jq_true.config.n_nodes
    out_suff = f"_K{n_nodes}_d{seed}"

    out_path = './'
    out_csv = os.path.join(out_path, "score" + out_suff + ".csv")
    results_df = None
    for i, sampling_relative_temp in enumerate(final_temps):
        print(f"scoring with rel temp: {sampling_relative_temp} ({i+1}/{len(final_temps)})")
        # set sampling temp
        if victree.q.t.temp != 1.:
            print("WARNING: qT temp is not 1., inference did not proceed with final full update.")
        victree.q.t.g_temp = sampling_relative_temp
        # save results
        results_df = evaluate_victree_to_df(jq_true, victree, dataset_id=seed, tree_enumeration=n_nodes < 7,
                                            sampling_relative_temp=sampling_relative_temp, df=results_df)

    results_df['extend_temp'] = extend_qt_temp
    results_df['gt_temp_mult'] = gt_temp_mult
    results_df['final_step'] = 1
    results_df['final_sample_temp'] = final_temps
    results_df.to_csv(out_csv, index=False)
    print(f"results saved in: {out_csv}")


if __name__ == '__main__':
    # read dataset path
    dat_path = sys.argv[1]
    extend_qt_temp = 0.8
    gt_temp_mult = 1.3
    final_temps = [1.]
    if len(sys.argv) > 2:
        final_temps = list(map(float, sys.argv[2].split(',')))
        print(f"setting final gt/qt temp to {final_temps}")

    params_re = re.compile(r'^.*/K(?P<K>\d+)M(?P<M>\d+)N(?P<N>\d+)/(?P<seed>\d+)\.png$')
    re_match = params_re.match(dat_path)
    K = int(re_match.group('K'))
    M = int(re_match.group('M'))
    N = int(re_match.group('N'))
    seed = int(re_match.group('seed'))
    print(f"using dataset: K {K} M {M} N {N}, dataset {seed}")

    jq_true, ad = gen_data(K, M, N, seed)
    print("inference started")
    victree = run_dataset(jq_true, ad, extend_qt_temp=extend_qt_temp, gt_temp_mult=gt_temp_mult)
    print("inference finished\ncomputing scores...")
    compute_scores(jq_true, victree, seed, final_temps)
