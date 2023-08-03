import logging
import os

import torch
import numpy as np

from inference.victree import VICTree
from utils.tree_utils import newick_from_eps_arr
from variational_distributions.joint_dists import VarTreeJointDist, FixedTreeJointDist
from utils.config import Config
from utils.data_handling import read_sc_data, load_h5_pseudoanndata, write_output, write_checkpoint_h5, DataHandler
from variational_distributions.var_dists import qMuTau, qEpsilonMulti, qPi, qCMultiChrom


def run(args):
    """
    Instantiate configuration object, variational distributions
    and observations. Run main inference algorithm.
    """
    # ---
    # Import data
    # ---
    data_handler = DataHandler(args.file_path)
    obs = data_handler.norm_reads

    # obs n_bins x n_cells matrix
    n_bins, n_cells = obs.shape
    logging.debug(f"file {args.file_path} read successfully [{n_bins} bins, {n_cells} cells]")

    # ---
    # Create configuration object
    # ---
    config = Config(n_nodes=args.n_nodes, n_states=args.n_states, n_cells=n_cells, chain_length=n_bins,
                    wis_sample_size=args.tree_sample_size, sieving_size=args.sieving[0], n_sieving_iter=args.sieving[1],
                    step_size=args.step_size, debug=args.debug, diagnostics=args.diagnostics, out_dir=args.out_dir,
                    n_run_iter=args.n_iter, elbo_rtol=args.r_tol, chromosome_indexes=data_handler.get_chr_idx())
    logging.debug(str(config))

    # ---
    # Instantiate all distributions with prior parameters
    # ---
    qmt = qMuTau(config, nu_prior=args.prior_mutau[0], lambda_prior=args.prior_mutau[1],
                 alpha_prior=args.prior_mutau[2], beta_prior=args.prior_mutau[3])
    qeps = qEpsilonMulti(config, alpha_prior=args.prior_eps[0], beta_prior=args.prior_eps[1])
    qpi = qPi(config, delta_prior=args.prior_pi)

    # if more chromosomes are provided, split into multiple qC sequences
    qc = None
    if config.chromosome_indexes:
        qc = qCMultiChrom(config)
    joint_q = VarTreeJointDist(config, obs, qc=qc, qmt=qmt, qeps=qeps, qpi=qpi)

    logging.info('initializing distributions..')
    joint_q.initialize()
    joint_q.z.initialize(z_init=args.z_init, obs=obs)
    joint_q.mt.initialize(method='data', obs=obs)
    joint_q.eps.initialize(method='uniform')

    # ---
    # Create copytree object and run inference
    # ---
    copy_tree = VICTree(config, joint_q, obs, data_handler=data_handler)

    logging.info('start inference')
    copy_tree.run(args=args)

    # ---
    # Save output to H5
    # ---
    write_output(copy_tree, os.path.join(config.out_dir, "out_" + str(copy_tree) + ".h5"))

    return copy_tree
