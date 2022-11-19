from typing import List, Tuple
import torch
import numpy as np

# TODO: read data and make config read data input specs
def read_sc_data(file_path: str) -> Tuple[List, List, torch.Tensor]:
    with open(file_path, 'r') as f:
        cell_names = f.readline().strip().split(" ")
        gene_ids = []
        obs_lst = []
        nlines = 0
        for line in f:
            lspl = line.strip().split(" ")
            gene_ids.append(lspl[0])
            obs_lst.append(list(map(int, lspl[1:])))
            nlines += 1

    obs = torch.tensor(obs_lst)


        # # name_matrix = np.genfromtxt(f, dtype=None,
        # #                          delimiter=" ", encoding=None)
        # name_matrix = np.loadtxt(f, dtype={'names': ('gene_id',) + tuple(cell_names), 
        #                                    'formats': ('U25',) + ('u8',) * len(cell_names)},
        #                          delimiter=" ", ndmin=2)
        # gene_ids = name_matrix['gene_id'].tolist()
        # obs = torch.tensor(name_matrix[1:])

    return cell_names, gene_ids, obs

            






