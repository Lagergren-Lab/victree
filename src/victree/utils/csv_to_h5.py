#!/usr/bin/env python3

from pathlib import Path
from anndata import read_csv
from sys import argv
import scgenome
import pandas as pd
import os

if __name__ == '__main__':
    data_dir = argv[1]
    print(f'reading files from: {data_dir}')
    cn_data = pd.read_csv(os.path.join(data_dir, 'ov2295_cell_cn.csv'),
                          dtype={
                              'cell_id': 'category',
                              'sample_id': 'category',
                              'library_id': 'category',
                              'chr': 'category',
                          })
    cn_data = cn_data[cn_data['chr'].isin(['1', '2', '3', '4', '5', '6', '7'])]

    metrics_data = pd.read_csv(os.path.join(data_dir, 'ov2295_cell_metrics.csv.gz'),
                               dtype={
                                   'cell_id': 'category',
                                   'sample_id': 'category',
                                   'library_id': 'category',
                               })

    scgenome.utils.union_categories([cn_data, metrics_data])
    cn_data['gc'] = 1.
    hmmcopy = scgenome.pp.convert_dlp_hmmcopy(metrics_data, cn_data)

    # save file
    hmmcopy.write(Path(os.path.join(data_dir, 'ov2295_chr1-7.h5')))
