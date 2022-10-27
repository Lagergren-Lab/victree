from networkx.algorithms.tree.recognition import is_arborescence
import torch
import networkx as nx

class TreeHMM(torch.nn.Module):

    def __init__(self, tree: nx.DiGraph):
        assert(is_arborescence(tree))

    def sample(self, T): pass
