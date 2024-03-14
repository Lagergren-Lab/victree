import os

import numpy as np
import networkx as nx
import torch

from utils.config import Config
from utils.tree_utils import tree_to_newick
from variational_distributions.var_dists import qT

sa501x3f_graph_path = "/Users/zemp/phd/notes/scilife/sa501x3f_graph_k12.txt"

if __name__=="__main__":
    matrix = np.loadtxt(sa501x3f_graph_path)
    qt = qT(Config(n_nodes=12))
    qt.initialize(method='matrix', matrix=torch.tensor(matrix))
    mst: nx.DiGraph = nx.maximum_spanning_arborescence(qt.weighted_graph, preserve_attrs=True)

    n = 50
    print(f"Computing PMF estimate over sample size of {n}")
    qt_pmf = qt.get_pmf_estimate(normalized=True, n=n, desc_sorted=True)
    for t_nwk, w in qt_pmf.items():
        print(f"{t_nwk}: {w}")

    print(f"MST: {tree_to_newick(mst)}, sum of weights [ log(\\tilde q(T)) ]:"
          f" {mst.size(weight='weight')}")
    for e in mst.edges():
        try:
            alt_graph = qt.weighted_graph.copy()
            alt_graph.remove_edge(*e)
            alt_mst = nx.maximum_spanning_arborescence(alt_graph,
                                                       preserve_attrs=True)
            alt_mst_nwk = tree_to_newick(alt_mst)
            print(f"alt-MST w/o ({e[0], e[1]}): {alt_mst_nwk}, sum of weights [ log(\\tilde q(T)) ]:"
                  f"{alt_mst.size(weight='weight')} {qt_pmf[alt_mst_nwk] if alt_mst_nwk in qt_pmf else 0.}")
        except nx.NetworkXException as nxe:
            print(f"MISS: cant find MST w/o ({e[0], e[1]})")
