from inference.copy_tree import CopyTree
from variational_distributions.variational_distribution import VariationalDistribution
from model.generative_model import GenerativeModel

def run(args):
    # TODO: write main code
    p = GenerativeModel()
    q = VariationalDistribution()
    copy_tree = CopyTree(p, q, q)
    
    copy_tree.run(args.n_iter)

    return copy_tree.elbo
