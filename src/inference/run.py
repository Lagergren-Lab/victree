import logging
import os

import torch

from inference.copy_tree import CopyTree, JointVarDist, VarDistFixedTree
from utils.config import Config
from utils.data_handling import read_sc_data, load_h5_anndata, write_output_h5


def run(args):
    fname, fext = os.path.splitext(args.file_path)
    if fext == '.txt':
        cell_names, gene_ids, obs = read_sc_data(args.file_path)
        obs = obs.float()
    elif fext == '.h5':
        full_data = load_h5_anndata(args.file_path)
        obs = torch.tensor(full_data['X'])
    else:
        raise FileNotFoundError(f"file extension not recognized: {fext}")

    n_bins, n_cells = obs.shape
    logging.debug(f"file {args.file_path} read successfully [{n_bins} bins, {n_cells} cells]")

    config = Config(chain_length=n_bins, n_cells=n_cells, n_nodes=args.K, n_states=args.A,
                    wis_sample_size=args.L, debug=args.debug)
    logging.debug(f"Config - n_nodes:  {args.K}, n_states:  {args.A}, n_tree_samples:  {args.L}")

    # instantiate all distributions
    joint_q = JointVarDist(config, obs)
    joint_q.initialize()
    copy_tree = CopyTree(config, joint_q, obs)
    
    copy_tree.run(args.n_iter)

    write_output_h5(copy_tree, args.out_path)

    return copy_tree
