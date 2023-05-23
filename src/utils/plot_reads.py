#!/usr/bin/env python3
import sys

import anndata
import matplotlib.pyplot as plt
import scgenome.plotting as pl

import simul
from utils.config import Config

if __name__ == '__main__':
    # h5_path = sys.argv[1]
    # ad = anndata.read_h5ad(h5_path)
    #

    data = simul.simulate_full_dataset(Config(4, n_states=10, n_cells=100, chain_length=200),
                                       eps_a=5., eps_b=100., dir_alpha=10.)
    ad = anndata.AnnData(X=data['raw'].numpy(), layers={
        'state': data['c'][data['z'], :].T.numpy(),
        'copy': data['obs'].numpy()
    })
    fig, ax = plt.subplots()
    plot_dict = pl.plot_cell_cn_matrix(ad, layer_name='state', ax=ax)
    fig.show()

