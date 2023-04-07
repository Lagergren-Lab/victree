import logging
import os

import torch
import numpy as np
import pickle
import yaml

from inference.copy_tree import CopyTree, JointVarDist, VarDistFixedTree
from utils import visualization_utils
from utils.config import Config
from utils.data_handling import read_sc_data, load_h5_anndata, write_output_h5


def write_diagnostics_to_numpy(diag_dict: dict[str, torch.Tensor], out_dir, config: Config):
    """
    Diagnostics files contain:
    - copy.npy (n_iter, K, M, A) single filtering probs
    - cell_assignment.npy (n_iter, N, K) z
    - pi.npy (n_iter, K) concentration factor
    - eps_a.npy, eps_b.npy (n_iter, K) beta params for epsilon
    - nu.npy, lmbda.npy, alpha.npy, beta.npy (n_iter, N) NormalGamma params
    - elbo.npy (n_iter, ) elbo

    Parameters
    ----------
    diag_dict
    out_dir
    """
    diag_dir = os.path.join(out_dir, f"diag_k{config.n_nodes}"
                                     f"a{config.n_states}"
                                     f"n{config.n_cells}"
                                     f"m{config.chain_length}")
    if not os.path.exists(diag_dir):
        os.makedirs(diag_dir)
    # copy numbers
    np.save(os.path.join(diag_dir, 'copy.npy'), diag_dict['C'].numpy())
    # cell assignment
    np.save(os.path.join(diag_dir, 'cell_assignment.npy'), diag_dict['Z'].numpy())
    # pi
    np.save(os.path.join(diag_dir, 'pi.npy'), diag_dict['pi'].numpy())
    # eps
    np.save(os.path.join(diag_dir, 'eps_a.npy'), diag_dict['eps_a'].numpy())
    np.save(os.path.join(diag_dir, 'eps_b.npy'), diag_dict['eps_b'].numpy())
    # mu-tau
    np.save(os.path.join(diag_dir, 'nu.npy'), diag_dict['nu'].numpy())
    np.save(os.path.join(diag_dir, 'lambda.npy'), diag_dict['lmbda'].numpy())
    np.save(os.path.join(diag_dir, 'alpha.npy'), diag_dict['alpha'].numpy())
    np.save(os.path.join(diag_dir, 'beta.npy'), diag_dict['beta'].numpy())
    # tree
    np.save(os.path.join(diag_dir, 'tree_samples.npy'), np.array(diag_dict['T']))
    np.save(os.path.join(diag_dir, 'tree_matrix.npy'), diag_dict['wG'].numpy())
    # elbo
    np.save(os.path.join(diag_dir, 'elbo.npy'), diag_dict['elbo'].numpy())
    # metadata
    with open(os.path.join(diag_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump(config.to_dict(), f)

    logging.info(f"diagnostics saved successfully in {diag_dir}")


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
    joint_q.z.initialize(method='uniform')
    # joint_q.z.initialize(method='kmeans', obs=obs)
    joint_q.mt.initialize(method='data', obs=obs)
    joint_q.eps.initialize(method='uniform')

    copy_tree = CopyTree(config, joint_q, obs)

    logging.info('start inference')
    copy_tree.run(args.n_iter)
    run_str = f'out_K{config.n_nodes}_A{config.n_states}_N{config.n_cells}_M{config.chain_length}'
    if args.diagnostics:
        write_diagnostics_to_numpy(copy_tree.diagnostics_dict, out_dir=args.out_dir, config=config)
        # file_dir = './output/'
        # file_name = f'diagnostics_K{config.n_nodes}_N{config.n_cells}_M{config.chain_length}_A{config.n_states}' \
        #             f'_iter{args.n_iter}'
        # file_name += f'_L{config.wis_sample_size}.pkl' if type(copy_tree.q) is JointVarDist else '.pkl'
        # file_path = os.path.join(file_dir, file_name)
        # with open(file_path, 'wb') as pickle_file:
        #     pickle.dump(copy_tree.diagnostics_dict, pickle_file)

    out_file = os.path.join(args.out_dir, run_str + '.h5')
    write_output_h5(copy_tree, out_file)

    return copy_tree
