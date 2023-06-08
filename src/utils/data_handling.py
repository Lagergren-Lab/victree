import logging
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


def edge_dict_to_matrix(a: dict, k: int):
    """
    zero pads the edges which are not in the dict keys
    k: size of matrix (num_nodes)
    """
    np_mat = np.zeros((k, k), dtype=np.float32)
    for uv, e in a.items():
        np_mat[uv] = e
    return np_mat


def write_output_h5(out_copytree, out_path):
    f = h5py.File(out_path, 'w')
    x_ds = f.create_dataset('X', data=out_copytree.obs.T)
    out_grp = f.create_group('result')

    graph_data = out_copytree.q.t.weighted_graph.edges.data('weight')
    graph_adj_matrix = nx.to_numpy_array(out_copytree.q.t.weighted_graph)
    k = graph_adj_matrix.shape[0]
    alpha_tensor = edge_dict_to_matrix(out_copytree.q.eps.alpha_dict, k)
    beta_tensor = edge_dict_to_matrix(out_copytree.q.eps.beta_dict, k)

    graph_weights = out_grp.create_dataset('graph', data=graph_adj_matrix)
    cell_assignment = out_grp.create_dataset('cell_assignment', data=out_copytree.q.z.pi)
    eps_alpha = out_grp.create_dataset('eps_alpha', data=alpha_tensor)
    eps_beta = out_grp.create_dataset('eps_beta', data=beta_tensor)

    mt_agg = torch.stack((out_copytree.q.mt.nu, out_copytree.q.mt.lmbda,
                          out_copytree.q.mt.alpha, out_copytree.q.mt.beta))

    mt = out_grp.create_dataset('mu_tau', data=mt_agg)

    f.close()
    logging.debug(f"results saved: {out_path}")


def write_checkpoint_h5(copytree, path=None):
    if copytree.cache_size > 0:
        if path is None:
            path = "./checkpoint_" + str(copytree) + ".h5"

        # append mode, so that if the file already exist, then the data is appended
        with h5py.File(path, 'a') as f:
            # for each of the individual q dist + the joint dist itself (e.g. to monitor joint_q.elbo)
            if len(f.keys()) == 0:
                # init h5 file
                for q in copytree.q.get_units() + [copytree.q]:
                    qlay = f.create_group(q.__class__.__name__)
                    for k in q.params_history.keys():
                        stacked_arr = np.stack(q.params_history[k], axis=0)
                        # init dset with unlimited number of iteration and fix other dims
                        ds = qlay.create_dataset(k, data=stacked_arr,
                                                 maxshape=(copytree.config.n_sieving_iter + copytree.config.n_run_iter + 1,
                                                           *stacked_arr.shape[1:]), chunks=True)
            else:
                # resize and append
                for q in copytree.q.get_units() + [copytree.q]:
                    qlay = f[q.__class__.__name__]
                    for k in q.params_history.keys():
                        stacked_arr = np.stack(q.params_history[k], axis=0)
                        ds = qlay[k]
                        ds.resize(ds.shape[0] + stacked_arr.shape[0], axis=0)
                        ds[-stacked_arr.shape[0]:] = stacked_arr

            # wipe cache
            for q in copytree.q.get_units() + [copytree.q]:
                q.params_history = {k: [] for k in q.params_history.keys()}

        logging.debug("checkpoint saved!")


def load_h5_anndata(file_path):
    return h5py.File(file_path, 'r')


def read_X_counts_from_h5(file_path):
    f = h5py.File(file_path, 'r')
    return f['X']


def read_hmmcopy_state_from_h5(file_path):
    f = h5py.File(file_path, 'r')
    return f['layers']['state']


def read_checkpoint(file_path):
    h5 = load_h5_anndata(file_path)
    data = {
        'elbo': h5['VarTreeJointDist']['elbo'][()],
        'copy': h5['qC']['single_filtering_probs'][()],
        'eps_alpha': h5['qEpsilonMulti']['alpha'][()],
        'eps_beta': h5['qEpsilonMulti']['beta'][()],
        'mt_nu': h5['qMuTau']['nu'][()],
        'mt_lmbda': h5['qMuTau']['lmbda'][()],
        'mt_alpha': h5['qMuTau']['alpha'][()],
        'mt_beta': h5['qMuTau']['beta'][()],
        'pi_cf': h5['qPi']['concentration_param'][()],
        't_sample_nwk': h5['qT']['trees_sample_newick'][()],
        't_sample_w': h5['qT']['trees_sample_weights'][()],
        't_mat': h5['qT']['weight_matrix'][()],
        'z_pi': h5['qZ']['pi'][()]
    }
    return data


def read_last_it_from_checkpoint(file_path):
    data = read_checkpoint(file_path)
    for k in data:
        data[k] = data[k][-1, ...]
    return data


def read_simul(file_path):
    h5 = load_h5_anndata(file_path)
    data = {
        'obs': h5['layers']['copy'][()],
        'copy': h5['gt']['copy'][()],
        'eps': h5['gt']['eps'][()],
        'z': h5['gt']['cell_assignment'][()],
        'mu': h5['gt']['mu'][()],
        'tau': h5['gt']['tau'][()],
        'pi': h5['gt']['pi'][()],
    }
    return data


def read_vi_gt(checkpoint_file, simul_file):
    vi = read_last_it_from_checkpoint(checkpoint_file)
    gt = read_simul(simul_file)
    return vi, gt
