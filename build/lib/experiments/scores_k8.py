#!/usr/bin/env python3

import os
import sys
import re

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from utils.data_handling import read_vi_gt
from utils.evaluation import best_mapping


def negative_ce(true_ohe, vi):
    return np.sum(true_ohe * np.log(vi))


if __name__=="__main__":
    # import checkpoint file at last iter
    check_folder = sys.argv[1]
    data_folder = sys.argv[2]
    # k5 uses k5epsprior and k8 dat
    checkpoints = []
    simuls = []
    for path, subdirs, files in os.walk(check_folder):
        for name in files:
            filename = os.path.join(path, name)
            if re.search(r'checkpoint.*\.h5', filename):
                print(f"found {filename}")
                checkpoints.append(filename)
                # find related simul
                simul_id = re.findall(r'\/(s[0-9]+)\/', filename)[0]
                for df in os.listdir(data_folder):
                    if re.search(simul_id, df):
                        print(f'found dataset! {df}')
                        simuls.append(os.path.join(data_folder, df))
                        break

    out = {
        'ari': [],
        'cnce': [],
        'znce': [],
        'elbo': [],
        'M': [],
        'K': []
    }
    for cf, sf in zip(checkpoints, simuls):
        try:
            vi_dat, gt_dat = read_vi_gt(cf, sf)
            # compute best mapping
            mapp, score = best_mapping(gt_dat['z'], vi_dat['z_pi'], with_score=True)
            print(f'best mapping: {mapp} -> score {score}')
            K, M, A = vi_dat['copy'].shape
            out['M'] = M
            out['K'] = K
            # read cell assignment
            vi_z = vi_dat['z_pi']  # (N, K)
            gt_z = gt_dat['z']

            # compute ari
            ari = adjusted_rand_score(gt_z, vi_z.argmax(axis=1))
            out['ari'].append(ari)
            print(f'ari: {ari}')

            gt_z_ohe = np.eye(K)[gt_z]  # (N, K)
            gt_z_ohemapp = gt_z_ohe[:, mapp]  # mapped
            # compute neg cross entropy
            z_neg_ce = negative_ce(gt_z_ohemapp, vi_z)
            out['znce'].append(z_neg_ce)
            print(f'negative cross-entropy for z {z_neg_ce}')

            # read copy number signal
            gt_c = gt_dat['copy'][mapp, :]  # (K, M)
            vi_c = vi_dat['copy']  # (K, M, A)
            gt_c_ohe = (np.arange(A) == gt_c[..., None]).astype(int)
            assert gt_c_ohe.shape == vi_c.shape
            # compute neg cross entropy on copy numbers
            c_neg_ce = negative_ce(gt_c_ohe, vi_c)
            out['cnce'].append(c_neg_ce)
            print(f"copy neg cross-entropy {c_neg_ce / K / M}")

            # compute elbo
            out['elbo'].append(vi_dat['elbo'])
            print(f"elbo {vi_dat['elbo']}")
        except Exception as e:
            print(e)

        df = pd.DataFrame(out)
        csv_out = os.path.join(check_folder, "scores.csv")
        print(f"csv saved in {csv_out}")
        df.to_csv(csv_out, index_label="idx")

