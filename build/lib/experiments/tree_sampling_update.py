#!/usr/bin/env python3

import time
import sys

import pandas as pd

import simul
from utils.config import Config, set_seed
from utils.tree_utils import tree_to_newick
from variational_distributions.var_dists import qEpsilonMulti, qC, qT

if __name__=='__main__':
    if len(sys.argv) < 2:
        raise ValueError("out dir not valid. Usage: script.py ./out.csv")
    out_csv = sys.argv[1]
    print(f"results will be saved in {out_csv}")

    klist = [5, 6, 7, 8, 9, 10]
    M = 200
    N = 1000
    L = 100
    llist = [100, 120, 200, 300, 400, 400]
    end_llist = [100, 100, 300, 400, 500, 500]
    n_run_iter_list = [20, 20, 60, 80, 100, 100]
    step_size = .2
    num_exp = 15

    ks = []
    probs = []
    percentiles = []
    gt_trees = []
    samp_times = []
    samp_sizes = []
    samp_sizes_upd = []
    upd_iters = []

    assert len(klist) == len(end_llist)

    for i in range(len(klist)):
        k = klist[i]
        l = end_llist[i]
        wis_ss = llist[i]
        n_run_iter = n_run_iter_list[i]
        print(f"running k={k}", end=": ")
        for j in range(num_exp):
            print(f"{j}", end=", ")
            set_seed(j * 123 + 4)
            # generate dataset
            config = Config(n_nodes=k, chain_length=M, n_cells=N, wis_sample_size=wis_ss, n_run_iter=20, step_size=step_size)
            data = simul.simulate_full_dataset(config=config, eps_a=40., eps_b=1000., lambda0=100., alpha0=500.,
                                               beta0=25., dir_delta=100.)
            # init distributions eps, c (from true) and T
            qeps: qEpsilonMulti = qEpsilonMulti(config, true_params={'eps': data['eps']})
            # qeps: qEpsilonMulti = qEpsilonMulti(config, alpha_prior=40., beta_prior=1000.).initialize()
            qc = qC(config, true_params={'c': data['c']})
            qt: qT = qT(config).initialize()
            # update qT for some iterations
            for _ in range(n_run_iter):
                qt.update(qc, qeps)
                # t, w = qt.get_trees_sample()
                # qeps.update(t, w, qc)

            gt_tree_newick = tree_to_newick(data['tree'])
            gt_trees.append(gt_tree_newick)

            # sample and track time
            start_sample = time.time()
            qt_pmf = qt.get_pmf_estimate(normalized=True, n=l)
            samp_times.append(time.time() - start_sample)

            sorted_newick = sorted(qt_pmf, key=qt_pmf.get, reverse=True)

            # current percentile accumulator
            perc = 0
            if gt_tree_newick not in sorted_newick:
                percentiles.append(1.)
                probs.append(0.)
            else:
                for dist_tree in sorted_newick:
                    if gt_tree_newick == dist_tree:
                        percentiles.append(perc)
                        probs.append(qt_pmf[dist_tree].item())
                        break
                    else:
                        perc += qt_pmf[dist_tree].item()

            ks.append(k)
            samp_sizes.append(l)
            samp_sizes_upd.append(wis_ss)
            upd_iters.append(n_run_iter)

        print(" done.")

    df = pd.DataFrame({
        'perc': percentiles,
        'gt_tree': gt_trees,
        'prob': probs,
        'K': ks,
        'time': samp_times,
        'sample_size_eval': samp_sizes,
        'sample_size_upd': samp_sizes_upd,
        'qt_upd_iters': upd_iters
    })
    df.to_csv(out_csv, index_label='idx')


