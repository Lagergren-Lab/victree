import logging

import networkx as nx
from Bio import Phylo

from inference.victree import VICTree
from variational_distributions.joint_dists import VarTreeJointDist, FixedTreeJointDist
from utils.config import Config
from utils.data_handling import DataHandler
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
                    n_run_iter=args.n_iter, elbo_rtol=args.r_tol, chromosome_indexes=data_handler.get_chr_idx(),
                    split=args.split)
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
    if args.tree_path:
        tree_nx = parse_newick(args, config)
        joint_q = FixedTreeJointDist(obs=obs, config=config, qc=qc, qeps=qeps, qpsi=qmt, qpi=qpi, T=tree_nx)
    else:
        joint_q = VarTreeJointDist(config, obs, qc=qc, qmt=qmt, qeps=qeps, qpi=qpi)

    logging.info('initializing distributions..')
    joint_q.initialize()
    joint_q.z.initialize(z_init=args.z_init, obs=obs)
    joint_q.mt.initialize(method='data', obs=obs)
    joint_q.eps.initialize(method='uniform')

    # ---
    # Create copytree object and run inference
    # ---
    victree = VICTree(config, joint_q, obs, data_handler=data_handler)

    logging.info('start inference')
    victree.run(args=args)

    # ---
    # Save output to H5
    # ---
    victree.write()

    return victree


def parse_newick(args, config):
    tree = Phylo.read(args.tree_path, 'newick')
    und_tree_nx = Phylo.to_networkx(tree)
    und_tree_nx = nx.convert_node_labels_to_integers(und_tree_nx)
    tree_nx = nx.DiGraph()
    tree_nx.add_edges_from(und_tree_nx.edges())
    assert tree_nx.number_of_nodes() == config.n_nodes, "newick tree does not match the number of nodes K"
    return tree_nx
