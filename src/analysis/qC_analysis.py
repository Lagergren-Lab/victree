import networkx as nx

from inference.victree import VICTree
from variational_distributions.joint_dists import FixedTreeJointDist


def selectTrees(qt):
    W = qt.weighted_graph
    return nx.maximum_spanning_arborescence(W)


def train_on_fixed_tree(victree: VICTree, n_iter: int):
    qmt = victree.q.mt
    qc = victree.q.c
    qz = victree.q.z
    qpi = victree.q.pi
    qT = victree.q.t
    qeps = victree.q.eps
    obs = victree.obs
    config = victree.config

    T = selectTrees(qT)
    q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, T, obs)

    victree_fixed_tree = VICTree(config, q, obs)
    victree_fixed_tree.run(n_iter)
    return victree_fixed_tree