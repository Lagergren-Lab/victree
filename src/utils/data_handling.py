import os.path
from typing import List, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import h5py
import networkx as nx


def read_sc_data(file_path: Union[str, Path]) -> Tuple[List, List, torch.Tensor]:
    with open(file_path, 'r') as f:
        cell_names = f.readline().strip().split(" ")
        gene_ids = []
        obs_lst = []
        nlines = 0
        for line in f:
            nlines += 1
            lspl = line.strip().split(" ")
            gene_ids.append(lspl[0])
            new_obs = list(map(int, lspl[1:]))
            if len(cell_names) != len(new_obs):
                err_msg = f"file format not valid: {file_path} has \
                {len(cell_names)} cells and {len(new_obs)} \
                reads at line {nlines} (gene_id {lspl[0]})"
                raise RuntimeError(err_msg)

            obs_lst.append(new_obs)

        obs = torch.tensor(obs_lst)
        return cell_names, gene_ids, obs


def dict_to_tensor(a: dict):
    a_tensor = torch.tensor([[u, v, w] for (u, v), w in a.items()])
    return a_tensor


def write_output_h5(out_copytree, out_path):
    f = h5py.File(out_path, 'w')
    x_ds = f.create_dataset('X', data=out_copytree.obs.T)
    out_grp = f.create_group('result')

    graph_data = out_copytree.q.t.weighted_graph.edges.data('weight')
    graph_adj_matrix = nx.to_numpy_array(out_copytree.q.t.weighted_graph)
    alpha_tensor = dict_to_tensor(out_copytree.q.eps.alpha_dict)
    beta_tensor = dict_to_tensor(out_copytree.q.eps.beta_dict)

    graph_weights = out_grp.create_dataset('graph', data=graph_adj_matrix)
    cell_assignment = out_grp.create_dataset('cell_assignment', data=out_copytree.q.z.pi)
    eps_alpha = out_grp.create_dataset('eps_alpha', data=alpha_tensor)
    eps_beta = out_grp.create_dataset('eps_beta', data=beta_tensor)

    mt_agg = torch.stack((out_copytree.q.mt.nu, out_copytree.q.mt.lmbda, out_copytree.q.mt.alpha, out_copytree.q.mt.beta))

    mt = out_grp.create_dataset('mu_tau', data=mt_agg)

    f.close()


def write_checkpoint_h5(copytree, path=None):
    if path is None:
        path = "./checkpoint_" + str(copytree) + ".h5"
    f = h5py.File(path, 'w')
    for q in copytree.q.get_units():
        # TODO: add q names to group params (one layer for each var dist?)
        qlay = f.create_group(q.__class__.__name__)
        for k in q.params_history.keys():
            stacked_arr = np.stack(q.params_history[k], axis=0)
            ds = qlay.create_dataset(k, data=stacked_arr)
    f.close()


def load_h5_anndata(file_path):
    return h5py.File(file_path, 'r')


def read_X_counts_from_h5(file_path):
    f = h5py.File(file_path, 'r')
    return f['X']


def read_hmmcopy_state_from_h5(file_path):
    f = h5py.File(file_path, 'r')
    return f['layers']['state']
