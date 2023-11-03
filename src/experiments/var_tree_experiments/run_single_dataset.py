#!/usr/bin/env python3
import os.path
import re
import sys

from inference.victree import make_input, VICTree
from utils.evaluation import sample_dataset_generation, evaluate_victree_to_df


def run_dataset(K, M, N, seed, extend_qt_temp=1., gt_temp_mult=1., final_step=False):
    print(f"running dat: K {K} M {M} N {N}, dataset {seed}")
    jq_true, ad = sample_dataset_generation(K, M, N, seed)
    n_nodes = jq_true.config.n_nodes
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
    config.gT_temp = gt_temp_mult * config.qT_temp
    config.temp_extend = extend_qt_temp
    victree = VICTree(config, jq, data_handler=dh, draft=True, elbo_rtol=1e-4)
    victree.run(100, final_step=final_step)

    # save results
    results_df = evaluate_victree_to_df(jq_true, victree, dataset_id=seed, tree_enumeration=n_nodes < 7)

    out_suff = f"temp{extend_qt_temp}" if extend_qt_temp != 1. else ""
    out_suff += f"gtm{gt_temp_mult}" if gt_temp_mult != 1. else ""
    out_suff += f"fs" if final_step else ""
    out_suff += f"d{seed}"

    out_path = './'
    out_csv = os.path.join(out_path, "score" + out_suff + ".csv")
    results_df['extend_temp'] = extend_qt_temp
    results_df['gt_temp_mult'] = gt_temp_mult
    results_df['final_step'] = int(final_step)
    results_df.to_csv(out_csv, index=False)
    print(f"results saved in: {out_csv}")


if __name__ == '__main__':
    # read dataset path
    dat_path = sys.argv[1]
    qt_temp_extend = 1.
    gt_temp_mult = 1.
    if len(sys.argv) > 2:
        qt_temp_extend = float(sys.argv[2])
        print(f"setting qt temp extend to {qt_temp_extend}")
    if len(sys.argv) > 3:
        gt_temp_mult = float(sys.argv[3])
        print(f"setting gT-temp = {gt_temp_mult} x qT-temp")

    params_re = re.compile(r'^.*/K(?P<K>\d+)M(?P<M>\d+)N(?P<N>\d+)/(?P<seed>\d+)\.png$')
    re_match = params_re.match(dat_path)
    run_dataset(int(re_match.group('K')), int(re_match.group('M')), int(re_match.group('N')),
                int(re_match.group('seed')), extend_qt_temp=qt_temp_extend, gt_temp_mult=gt_temp_mult,
                final_step=True)
