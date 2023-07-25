import logging
import os
from typing import List, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import h5py
import networkx as nx
import anndata
from anndata._io.utils import AnnDataReadError

#from inference.victree import VICTree
from utils.tree_utils import newick_from_eps_arr


class DataHandler:

    def __init__(self, file_path: str):
        """
        Reads the file in the specified path and allows for multiple data formats.
        The supported formats are:
            - AnnData files, with bins metadata in adata.var.chr (real data)
            - AnnData-like H5 files which only contain /layers/copy matrix (mainly for simulated data)
            - txt file with tabular reads
        Parameters
        ----------
        file_path: str, absolute path of the file
        """
        self._read_multiple_sources(file_path)

    def _read_multiple_sources(self, file_path: str):
        self.norm_reads = None
        self.chr_pd = None
        self.start = None
        self.end = None

        fname, fext = os.path.splitext(file_path)
        if fext == '.txt':
            # handle both simple tables in text files
            cell_names, gene_ids, obs = read_sc_data(file_path)
            obs = obs.float()
        elif fext in {'.h5', '.h5ad'}:
            try:
                # actual AnnData format
                logging.debug("reading anndata file")
                ann_dataset = anndata.read_h5ad(file_path)
                ann_dataset = _remove_nans(ann_dataset)
                ann_dataset = _sort_anndata(ann_dataset)
                obs = torch.tensor(ann_dataset.layers['copy'].T, dtype=torch.float)

                # pandas categorical for chromosomes
                self.chr_pd = ann_dataset.var.chr
                self.start = ann_dataset.var.start
                self.end = ann_dataset.var.end

            except AnnDataReadError as ae:
                logging.debug("anndata read failed. reading pseudo-anndata h5 file")
                # and binary H5 files in pseudo-anndata format
                full_data = load_h5_pseudoanndata(file_path)
                if 'gt' in full_data.keys():
                    # H5 file can contain ground truth for post-analysis (e.g. synthetic ds)
                    logging.debug(f"gt tree: {newick_from_eps_arr(full_data['gt']['eps'][...])}")

                obs = torch.tensor(np.array(full_data['layers']['copy']), dtype=torch.float).T
        else:
            raise FileNotFoundError(f"file extension not recognized: {fext}")

        self.norm_reads = obs
        self.n_bins, self.n_cells = obs.shape

    def _clean_dataset(self):
        if torch.any(torch.isnan(self.norm_reads)):
            # TODO: temporary solution for nans in data. 1D interpolation should work better
            obs = torch.nan_to_num(self.norm_reads, nan=2.0)

    def get_chr_idx(self):
        indexes = []
        if self.chr_pd is None:
            logging.warning("Getting indices for data with no chromosome specs."
                            " Execution will proceed on one single chain")
        else:
            # TODO: improve
            # get indices over the obs matrix when chromosome changes
            curr_chr = self.chr_pd[0]
            for i, c in enumerate(self.chr_pd):
                if c != curr_chr:
                    indexes.append(i)
                    curr_chr = c

        return indexes


def _sort_anndata(ann_dataset):
    if not ann_dataset.var.chr.cat.ordered:
        ord_chr = [str(c) for c in range(1, 23)] + ['X', 'Y']
        ann_dataset.var.chr = ann_dataset.var.chr.cat.reorder_categories(ord_chr, ordered=True)
    return ann_dataset[:, ann_dataset.var.sort_values(['chr', 'start']).index].copy()


def _remove_nans(ann_dataset: anndata.AnnData) -> anndata.AnnData:
    """
    Remove bins corresponding to NaNs in the 'copy' layer
    """
    # TODO: implement different NaN strategies
    ann_dataset = ann_dataset[:, ~ np.isnan(ann_dataset.layers['copy']).any(axis=0)].copy()
    return ann_dataset


def read_sc_data(file_path: Union[str, Path]) -> Tuple[List, List, torch.Tensor]:
    # FIXME: obsolete function, remove or adapt to new inputs
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
    mt_agg = torch.stack((out_copytree.q.mt.nu, out_copytree.q.mt.lmbda,
                          out_copytree.q.mt.alpha, out_copytree.q.mt.beta))


    copy_number = out_grp.create_dataset('copy_number', data=out_copytree.q.c.single_filtering_probs)
    graph_weights = out_grp.create_dataset('graph', data=graph_adj_matrix)
    cell_assignment = out_grp.create_dataset('cell_assignment', data=out_copytree.q.z.pi)
    eps_alpha = out_grp.create_dataset('eps_alpha', data=alpha_tensor)
    eps_beta = out_grp.create_dataset('eps_beta', data=beta_tensor)
    mt = out_grp.create_dataset('mu_tau', data=mt_agg)

    # store trees in a separate group
    qt_pmf = out_copytree.q.t.get_pmf_estimate(normalized=True, desc_sorted=True)
    trees_grp = out_grp.create_group('trees')
    newick_ds = trees_grp.create_dataset('newick', data=np.array(list(qt_pmf.keys()), dtype='S'))
    tree_weight_ds = trees_grp.create_dataset('weight', data=np.array(list(qt_pmf.values())))

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


def load_h5_pseudoanndata(file_path):
    return h5py.File(file_path, 'r')


def read_X_counts_from_h5(file_path):
    f = h5py.File(file_path, 'r')
    return f['X']


def read_hmmcopy_state_from_h5(file_path):
    f = h5py.File(file_path, 'r')
    return f['layers']['state']


def read_checkpoint(file_path):
    h5 = load_h5_pseudoanndata(file_path)
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
    h5 = load_h5_pseudoanndata(file_path)
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


def load_victree_model():
    victree = VICTree()
    return victree