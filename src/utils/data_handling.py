from typing import List, Tuple, Union
from pathlib import Path
import torch
import h5py


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


def load_h5_anndata(file_path):
    return h5py.File(file_path, 'r')


def read_X_counts_from_h5(file_path):
    f = h5py.File(file_path, 'r')
    return f['X']


def read_hmmcopy_state_from_h5(file_path):
    f = h5py.File(file_path, 'r')
    return f['layers']['state']
