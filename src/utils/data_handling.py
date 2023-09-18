import logging
import os
import pathlib
from pathlib import Path

import numpy as np
import torch
import h5py
import networkx as nx
import anndata

from utils.tree_utils import newick_from_eps_arr, tree_to_newick


class DataHandler:

    def __init__(self, file_path: str | None = None, adata: anndata.AnnData | None = None):
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
        self.chr_df = None
        if adata is not None:
            self._adata = adata
            self.chr_df = adata.var[['chr', 'start', 'end']].reset_index()
            self._obs = torch.tensor(adata.layers['copy']).T.float()
        elif file_path is not None:
            self._adata, self._obs = self._read_multiple_sources(file_path)
        else:
            raise ValueError("provide either file path or anndata object")

    def _read_multiple_sources(self, file_path: str) -> (anndata.AnnData, torch.Tensor):
        adata = anndata.AnnData()
        self.chr_df = None

        fname, fext = os.path.splitext(file_path)

        if fext in {'.h5', '.h5ad'}:
            try:
                # actual AnnData format
                logging.debug("reading anndata file")
                ann_dataset = anndata.read_h5ad(file_path)

                ann_dataset = _sort_anndata(ann_dataset)
                ann_dataset = _impute_nans(ann_dataset, method='ignore')
                obs = torch.tensor(ann_dataset.layers['copy'].T, dtype=torch.float)

                # pandas categorical for chromosomes
                self.chr_df = ann_dataset.var[['chr', 'start', 'end']].reset_index()
                # original anndata file can potentially be kept with all its layers
                # but for simplicity, a new clean anndata file is instantiated instead
                # (gave issues with some datasets)
                # adata = ann_dataset

                adata = anndata.AnnData(obs.T.numpy())
                adata.var = self.chr_df

            except Exception as ae:  # Couldn't load module for AnnDataReadError
                logging.debug("anndata read failed. reading pseudo-anndata h5 file")
                # and binary H5 files in pseudo-anndata format
                full_data = load_h5_pseudoanndata(file_path)
                if 'gt' in full_data.keys():
                    # H5 file can contain ground truth for post-analysis (e.g. synthetic ds)
                    logging.debug(f"gt tree: {newick_from_eps_arr(full_data['gt']['eps'][...])}")

                obs = torch.tensor(np.array(full_data['layers']['copy']), dtype=torch.float).T
                adata.X = obs.T.numpy()
        else:
            raise FileNotFoundError(f"file extension not recognized: {fext}")

        return adata, obs

    def get_anndata(self) -> anndata.AnnData:
        return self._adata

    @property
    def norm_reads(self) -> torch.Tensor:
        return self._obs

    def _clean_dataset(self):
        if torch.any(torch.isnan(self.norm_reads)):
            # TODO: temporary solution for nans in data. 1D interpolation should work better
            obs = torch.nan_to_num(self.norm_reads, nan=2.0)

    def get_chr_idx(self):
        indices = []
        if self.chr_df is None:
            logging.warning("Getting indices for data with no chromosome specs."
                            " Execution will proceed on one single chain")
        else:
            indices = self.chr_df.index[self.chr_df['chr'].ne(self.chr_df['chr'].shift())].to_list()[1:]

        return indices


def _sort_anndata(ann_dataset):
    if not ann_dataset.var.chr.cat.ordered:
        ord_chr = [str(c) for c in range(1, 23)] + ['X', 'Y']
        ann_dataset.var.chr = ann_dataset.var.chr.cat.reorder_categories(ord_chr, ordered=True)
    return ann_dataset[:, ann_dataset.var.sort_values(['chr', 'start']).index].copy()


def _impute_nans(ann_dataset: anndata.AnnData, method: str = 'ignore') -> anndata.AnnData:
    """
    Remove bins corresponding to NaNs in the 'copy' layer
    method = ['ignore', 'remove', 'fill']
    """
    nan_sites_count = np.isnan(ann_dataset.layers['copy']).any(axis=0).sum()
    if nan_sites_count > 0:
        logging.debug(f"found {nan_sites_count} sites with nan values. proceeding with method `{method}`")

        if method == 'remove':
            ann_dataset = ann_dataset[:, ~ np.isnan(ann_dataset.layers['copy']).any(axis=0)]
            # drop chromosomes with just one site
            filter_df = ann_dataset.var['chr'].value_counts() < 2
            unit_chr_list = filter_df[filter_df].index.to_list()
            ann_dataset = ann_dataset[:, ann_dataset.var['chr'].isin(unit_chr_list)].copy()
            logging.debug(f"removed chromosome(s): {unit_chr_list}")
        elif method == 'fill':
            ann_dataset.layers['copy'][:, np.isnan(ann_dataset.layers['copy']).any(axis=0)] = 2.0
        elif method == 'ignore':
            pass
        else:
            raise NotImplementedError(f"method {method} for imputing nans is not available")
    return ann_dataset


def dict_to_tensor(a: dict):
    a_tensor = torch.tensor([[u, v, w] for (u, v), w in a.items()])
    return a_tensor


def edge_dict_to_matrix(a: dict, k: int) -> np.ndarray:
    """
    zero pads the edges which are not in the dict keys
    k: size of matrix (num_nodes)
    """
    np_mat = np.zeros((k, k), dtype=np.float32)
    for uv, e in a.items():
        np_mat[uv] = e
    return np_mat


def write_output(victree, out_path, anndata: bool = True):
    # TODO: move to victree object method
    if os.path.exists(out_path):
        logging.warning("overwriting existing output...")

    if not anndata:
        write_output_h5(victree, out_path)
    else:
        write_output_anndata(victree, out_path)
    logging.debug(f"results successfully saved: {out_path}")


def write_output_anndata(victree, out_path):
    adata: anndata.AnnData = victree.data_handler.get_anndata()

    # prepare variables
    # argmax cell assignment as a n_cell long vector
    top_z = torch.argmax(victree.q.z.pi, dim=1)

    # layers copy number (n_cells, n_sites)
    adata.layers['victree-cn-viterbi'] = victree.q.c.get_viterbi()[top_z].numpy()
    adata.layers['victree-cn-marginal'] = victree.q.c.single_filtering_probs[top_z].numpy()

    # obs - mu/tau dist, estimated clone (n_cells,)
    adata.obs['victree-mu'] = victree.q.mt.nu.numpy()
    adata.obs['victree-mt-lambda'] = victree.q.mt.lmbda.numpy()
    adata.obs['victree-mt-alpha'] = victree.q.mt.alpha.numpy()
    adata.obs['victree-mt-beta'] = victree.q.mt.beta.numpy()
    adata.obs['victree-tau'] = victree.q.mt.exp_tau().numpy()
    adata.obs['victree-clone'] = top_z.numpy()

    # obsm - clone probs (n_cells, ...)
    adata.obsm['victree-clone-probs'] = victree.q.z.pi.numpy()

    # varm - clonal copy number (single and pair) probs (n_sites, ...)
    adata.varm['victree-cn-sprobs'] = torch.permute(victree.q.c.single_filtering_probs, (1, 0, 2)).numpy()
    adata.varm['victree-cn-pprobs'] = torch.permute(victree.q.c.get_padded_cfp(), (1, 0, 2, 3)).numpy()

    # unstructured - tree, eps
    k = victree.config.n_nodes
    alpha_tensor = edge_dict_to_matrix(victree.q.eps.alpha_dict, k)
    beta_tensor = edge_dict_to_matrix(victree.q.eps.beta_dict, k)

    adata.uns['victree-eps-alpha'] = alpha_tensor
    adata.uns['victree-eps-beta'] = beta_tensor

    if hasattr(victree.q, "T"):
        # FixedTreeJointDist
        adata.uns['victree-tree-newick'] = np.array([tree_to_newick(victree.q.T)], dtype='S')
    else:
        qt_pmf = victree.q.t.get_pmf_estimate(normalized=True, desc_sorted=True)
        adata.uns['victree-tree-graph'] = nx.to_numpy_array(victree.q.t.weighted_graph)
        adata.uns['victree-tree-newick'] = np.array(list(qt_pmf.keys()), dtype='S')
        adata.uns['victree-tree-probs'] = np.array(list(qt_pmf.values()))

    adata.write_h5ad(Path(out_path))


def write_output_h5(victree, out_path):
    f = h5py.File(out_path, 'w')
    x_ds = f.create_dataset('X', data=victree.obs.T)
    out_grp = f.create_group('result')

    if hasattr(victree.q, "t"):
        graph_data = victree.q.t.weighted_graph.edges.data('weight')
        graph_adj_matrix = nx.to_numpy_array(victree.q.t.weighted_graph)
    k = victree.config.n_nodes
    alpha_tensor = edge_dict_to_matrix(victree.q.eps.alpha_dict, k)
    beta_tensor = edge_dict_to_matrix(victree.q.eps.beta_dict, k)
    mt_agg = torch.stack((victree.q.mt.nu, victree.q.mt.lmbda,
                          victree.q.mt.alpha, victree.q.mt.beta))

    copy_number = out_grp.create_dataset('cn_marginal', data=victree.q.c.single_filtering_probs.numpy())
    cn_viterbi = out_grp.create_dataset('cn_viterbi', data=victree.q.c.get_viterbi().numpy())
    graph_weights = out_grp.create_dataset('graph', data=graph_adj_matrix) if hasattr(victree.q, "t") else None
    cell_assignment = out_grp.create_dataset('cell_assignment', data=victree.q.z.pi.numpy())
    eps_alpha = out_grp.create_dataset('eps_alpha', data=alpha_tensor)
    eps_beta = out_grp.create_dataset('eps_beta', data=beta_tensor)
    mt = out_grp.create_dataset('mu_tau', data=mt_agg.numpy())

    # store trees in a separate group
    if hasattr(victree.q, "t"):
        qt_pmf = victree.q.t.get_pmf_estimate(normalized=True, desc_sorted=True)
        trees_grp = out_grp.create_group('trees')
        newick_ds = trees_grp.create_dataset('newick', data=np.array(list(qt_pmf.keys()), dtype='S'))
        tree_weight_ds = trees_grp.create_dataset('weight', data=np.array(list(qt_pmf.values())))

    f.close()

def write_checkpoint_h5(self, path=None):
    pass

def load_h5_pseudoanndata(file_path):
    return h5py.File(file_path, 'r')


def read_X_counts_from_h5(file_path):
    f = h5py.File(file_path, 'r')
    return f['X']


def read_hmmcopy_state_from_h5(file_path):
    f = h5py.File(file_path, 'r')
    return f['layers']['state']


def read_checkpoint(file_path):
    return load_h5_pseudoanndata(file_path)


def read_last_it_from_checkpoint(file_path):
    data = read_checkpoint(file_path)
    last_it_data = {}
    # get groups
    for group in data:
        # datasets
        last_it_data[group] = {}
        for ds in data[group]:
            last_it_data[group][ds] = data[group][ds][-1]
    return last_it_data


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


def create_analysis_output_catalog(analysis_function_path, base_dir):
    path = base_dir + "/" + analysis_function_path
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        print("Dir already exists. Risk of overwriting contents.")
    return path
