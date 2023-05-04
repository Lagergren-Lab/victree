import logging
import os

import networkx as nx
import torch
import numpy as np
import pickle
import yaml

from inference.copy_tree import CopyTree
from utils.tree_utils import newick_from_eps_arr
from variational_distributions.joint_dists import VarTreeJointDist, FixedTreeJointDist
from utils.config import Config
from utils.data_handling import read_sc_data, load_h5_anndata, write_output_h5, write_checkpoint_h5


def run(args):
    fname, fext = os.path.splitext(args.file_path)
    if fext == '.txt':
        cell_names, gene_ids, obs = read_sc_data(args.file_path)
        obs = obs.float()
    elif fext == '.h5':
        full_data = load_h5_anndata(args.file_path)
        if 'gt' in full_data.keys():
            logging.debug(f"gt tree: {newick_from_eps_arr(full_data['gt']['eps'][...])}")

        obs = torch.tensor(np.array(full_data['layers']['copy']), dtype=torch.float).T
        # FIXME: temporary solution for nans in data. 1D interpolation should work best
        if torch.any(torch.isnan(obs)):
            obs = torch.nan_to_num(obs, nan=2.0)
    else:
        raise FileNotFoundError(f"file extension not recognized: {fext}")

    n_bins, n_cells = obs.shape
    logging.debug(f"file {args.file_path} read successfully [{n_bins} bins, {n_cells} cells]")

    config = Config(n_nodes=args.n_nodes, n_states=args.n_states, n_cells=n_cells, chain_length=n_bins,
                    wis_sample_size=args.tree_sample_size, sieving_size=args.sieving[0], n_sieving_iter=args.sieving[1],
                    step_size=args.step_size, debug=args.debug, diagnostics=args.diagnostics)
    logging.debug(str(config))

    # instantiate all distributions
    joint_q = VarTreeJointDist(config, obs)
    logging.info('initializing distributions..')
    joint_q.initialize()
    joint_q.z.initialize(method='random')
    # joint_q.z.initialize(method='kmeans', obs=obs)
    joint_q.mt.initialize(method='data', obs=obs)
    joint_q.eps.initialize(method='uniform')

    copy_tree = CopyTree(config, joint_q, obs)

    logging.info('start inference')
    copy_tree.run(args.n_iter)
    run_str = f'out_K{config.n_nodes}_A{config.n_states}_N{config.n_cells}_M{config.chain_length}'

    if args.diagnostics:

        ## first version
        # file_dir = './output/'
        # file_name = f'diagnostics_K{config.n_nodes}_N{config.n_cells}_M{config.chain_length}_A{config.n_states}' \
        #             f'_iter{args.n_iter}'
        # file_name += f'_L{config.wis_sample_size}.pkl' if type(copy_tree.q) is JointVarDist else '.pkl'
        # file_path = os.path.join(file_dir, file_name)
        # with open(file_path, 'wb') as pickle_file:
        #     pickle.dump(copy_tree.diagnostics_dict, pickle_file)

        ## second version
        # write_diagnostics_to_numpy(copy_tree.q.diagnostics_dict, out_dir=args.out_dir, config=config)
        ## third version
        write_checkpoint_h5(copy_tree, path=os.path.join(args.out_dir, "checkpoint_" + str(copy_tree) + ".h5"))

    out_file = os.path.join(args.out_dir, run_str + '.h5')
    write_output_h5(copy_tree, out_file)
    logging.info(f"results saved: {out_file}")

    return copy_tree
