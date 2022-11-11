import torch

from inference.copy_tree import CopyTree, JointVarDist
from utils.config import Config
from variational_distributions.q_T import q_T
from variational_distributions.q_Z import qZ
from variational_distributions.q_epsilon import qEpsilon
from variational_distributions.variational_distribution import VariationalDistribution
from model.generative_model import GenerativeModel
from variational_distributions.variational_hmm import CopyNumberHmm
from variational_distributions.variational_normal import qMuTau

def run(args):
    # TODO: write main code
    config = Config()
    obs = torch.ones((config.n_states, config.n_cells))
    # obs = read_data()
    p = GenerativeModel()
    
    # instantiate all distributions 
    qc = CopyNumberHmm(config)
    qz = qZ(config)
    qt = q_T(config)
    qeps = qEpsilon(config)
    qmt = qMuTau(config)

    q = JointVarDist(config, qc, qz, qt, qeps, qmt, obs)
    copy_tree = CopyTree(config, p, q, obs)
    
    copy_tree.run(args.n_iter)

    return copy_tree.elbo
