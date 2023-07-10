import logging
import os

import torch
import numpy as np

from inference.copy_tree import CopyTree
from utils.tree_utils import newick_from_eps_arr
from variational_distributions.joint_dists import VarTreeJointDist, FixedTreeJointDist
from utils.config import Config
from utils.data_handling import read_sc_data, load_h5_anndata, write_output_h5, write_checkpoint_h5
from variational_distributions.var_dists import qMuTau, qEpsilonMulti, qPi


def run(args):
    """
    Instantiate configuration object, variational distributions
    and observations. Run main inference algorithm.
    """
    # ---
    # Import data
    # ---
    fname, fext = os.path.splitext(args.file_path)
    if fext == '.txt':
        # handle both simple tables in text files
        cell_names, gene_ids, obs = read_sc_data(args.file_path)
        obs = obs.float()
    elif fext == '.h5':
        # and binary H5 files in anndata format
        full_data = load_h5_anndata(args.file_path)
        if 'gt' in full_data.keys():
            # H5 file can contain ground truth for post-analysis (e.g. synthetic ds)
            logging.debug(f"gt tree: {newick_from_eps_arr(full_data['gt']['eps'][...])}")

        obs = torch.tensor(np.array(full_data['layers']['copy']), dtype=torch.float).T
        if torch.any(torch.isnan(obs)):
            # TODO: temporary solution for nans in data. 1D interpolation should work better
            obs = torch.nan_to_num(obs, nan=2.0)
    else:
        raise FileNotFoundError(f"file extension not recognized: {fext}")

    n_bins, n_cells = obs.shape
    logging.debug(f"file {args.file_path} read successfully [{n_bins} bins, {n_cells} cells]")

    # ---
    # Create configuration object
    # ---
    config = Config(n_nodes=args.n_nodes, n_states=args.n_states, n_cells=n_cells, chain_length=n_bins,
                    wis_sample_size=args.tree_sample_size, sieving_size=args.sieving[0], n_sieving_iter=args.sieving[1],
                    step_size=args.step_size, debug=args.debug, diagnostics=args.diagnostics, out_dir=args.out_dir,
                    n_run_iter=args.n_iter, elbo_rtol=args.r_tol)
    logging.debug(str(config))

    # ---
    # Instantiate all distributions with prior parameters
    # ---
    qmt = qMuTau(config, nu_prior=args.prior_mutau[0], lambda_prior=args.prior_mutau[1],
                 alpha_prior=args.prior_mutau[2], beta_prior=args.prior_mutau[3])
    qeps = qEpsilonMulti(config, alpha_prior=args.prior_eps[0], beta_prior=args.prior_eps[1])
    qpi = qPi(config, delta_prior=args.prior_pi)

    joint_q = VarTreeJointDist(config, obs, qmt=qmt, qeps=qeps, qpi=qpi)

    logging.info('initializing distributions..')
    joint_q.initialize()
    joint_q.z.initialize(z_init=args.z_init, obs=obs)
    joint_q.mt.initialize(method='data', obs=obs)
    joint_q.eps.initialize(method='uniform')

    # ---
    # Create copytree object and run inference
    # ---
    copy_tree = CopyTree(config, joint_q, obs)

    logging.info('start inference')
    copy_tree.run(args)

    # ---
    # Save output to H5
    # ---
    write_output_h5(copy_tree, os.path.join(args.out_dir, "out_" + str(copy_tree) + ".h5"))

    return copy_tree
