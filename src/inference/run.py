import logging

from inference.copy_tree import CopyTree, JointVarDist, VarDistFixedTree
from utils.config import Config
from utils.data_handling import read_sc_data
from utils.tree_utils import generate_fixed_tree
from variational_distributions.var_dists import qT, qEpsilon, qEpsilonMulti, qMuTau, qPi, qZ, qC
from model.generative_model import GenerativeModel

def run(args):
    # TODO: write main code
    cell_names, gene_ids, obs = read_sc_data(args.filename)
    n_genes, n_cells = obs.shape
    obs = obs.float()
    logging.debug(f"file {args.filename} read successfully [{n_genes} genes, {n_cells} cells]")

    config = Config(chain_length=n_genes, n_cells=n_cells, n_nodes=args.K, n_states=args.A, wis_sample_size=args.L)
    logging.debug(f"Config - n_nodes:  {args.K}, n_states:  {args.A}, n_tree_samples:  {args.L}")
    p = GenerativeModel(config)
    
    # instantiate all distributions 
    qc = qC(config)
    qz = qZ(config)
    qt = qT(config)
    qeps = qEpsilonMulti(config)
    qmt = qMuTau(config)
    qpi = qPi(config)

    # q = JointVarDist(config, qc, qz, qt, qeps, qmt, qpi, obs)
    tree = generate_fixed_tree(config.n_nodes)
    q = VarDistFixedTree(config, qc, qz, qeps, qmt, qpi, tree, obs)
    copy_tree = CopyTree(config, p, q, obs)
    
    copy_tree.run(args.n_iter)

    return copy_tree.elbo
