import logging
import os

import torch
import numpy as np
import pickle

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
        if 'gt' in full_data.keys():
            logging.debug(f"gt tree: {full_data['gt']['tree']}")

        obs = torch.tensor(np.array(full_data['X']))
    else:
        raise FileNotFoundError(f"file extension not recognized: {fext}")

    n_bins, n_cells = obs.shape
    logging.debug(f"file {args.file_path} read successfully [{n_bins} bins, {n_cells} cells]")

    config = Config(chain_length=n_bins, n_cells=n_cells, n_nodes=args.K, n_states=args.A,
                    wis_sample_size=args.L, debug=args.debug, step_size=args.step_size, diagnostics=args.debug)
    logging.debug(f"Config - n_nodes:  {args.K}, n_states:  {args.A}, n_tree_samples:  {args.L}")

    # instantiate all distributions
    joint_q = JointVarDist(config, obs)
    joint_q.initialize()
    copy_tree = CopyTree(config, joint_q, obs)
    
    copy_tree.run(args.n_iter)
    if args.debug:
        with open('./output/diagnostics.pkl', 'wb') as pickle_file:
            pickle.dump(copy_tree.diagnostics_dict, pickle_file)

    write_output_h5(copy_tree, args.out_path)

    return copy_tree
