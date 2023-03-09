import logging
import os

import torch
import numpy as np
import pickle

from inference.copy_tree import CopyTree, JointVarDist, VarDistFixedTree
from utils import visualization_utils
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
            logging.debug(f"gt tree: {np.array(full_data['gt']['tree'])}")

        obs = torch.tensor(np.array(full_data['X']), dtype=torch.float).T
    else:
        raise FileNotFoundError(f"file extension not recognized: {fext}")

    n_bins, n_cells = obs.shape
    logging.debug(f"file {args.file_path} read successfully [{n_bins} bins, {n_cells} cells]")

    config = Config(chain_length=n_bins, n_cells=n_cells, n_nodes=args.K, n_states=args.A,
                    wis_sample_size=args.L, debug=args.debug, step_size=args.step_size,
                    diagnostics=args.diagnostics)
    logging.debug(str(config))

    # instantiate all distributions
    joint_q = JointVarDist(config, obs)
    logging.info('initializing distributions..')
    joint_q.initialize()
    joint_q.z.initialize(method='kmeans', obs=obs)
    joint_q.mt.initialize(method='data', obs=obs)

    copy_tree = CopyTree(config, joint_q, obs)

    logging.info('start inference')
    copy_tree.run(args.n_iter)
    if args.diagnostics:
        file_dir = './output/'
        file_name = f'diagnostics_K{config.n_nodes}_N{config.n_cells}_M{config.chain_length}_A{config.n_states}' \
                    f'_iter{args.n_iter}'
        file_name += f'_L{config.wis_sample_size}.pkl' if type(copy_tree.q) is JointVarDist else '.pkl'
        file_path = os.path.join(file_dir, file_name)
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(copy_tree.diagnostics_dict, pickle_file)

    write_output_h5(copy_tree, args.out_path)

    return copy_tree
