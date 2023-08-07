import copy
import logging

import networkx as nx

from inference.victree import VICTree
from variational_distributions.joint_dists import FixedTreeJointDist


def selectTrees(qt):
    W = qt.weighted_graph
    T = nx.maximum_spanning_arborescence(W)
    logging.info(f'Selected tree: {T.edges}')
    return T


def train_on_fixed_tree(victree: VICTree, n_iter: int):
    victree_copy = copy.deepcopy(victree)
    qmt = victree_copy.q.mt
    qc = victree_copy.q.c
    qz = victree_copy.q.z
    qpi = victree_copy.q.pi
    qT = victree_copy.q.t
    qeps = victree_copy.q.eps
    obs = victree_copy.obs
    config = victree_copy.config

    T = selectTrees(qT)
    q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, T, obs)

    victree_fixed_tree = VICTree(config, q, obs)
    victree_fixed_tree.run(n_iter)
    return victree_fixed_tree