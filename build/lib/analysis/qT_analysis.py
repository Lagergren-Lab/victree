import os

import matplotlib
import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

from inference.victree import VICTree
from utils import tree_utils, factory_utils, data_handling
from utils.config import Config
from variational_distributions.var_dists import qT


def edge_probability_analysis(q_T: qT, L, true_tree=None, best_perm=None):
    """
    Given a trained qT, analyzes the edges in the trees sampled trees.
    """
    matplotlib.use('module://backend_interagg')

    # T comparisons
    N, M, K, A = (q_T.config.n_cells, q_T.config.chain_length, q_T.config.n_nodes, q_T.config.n_states)
    T_list_seed, w_T_list_seed = q_T.get_trees_sample(sample_size=L)
    unique_seq, unique_seq_idx, multiplicity = tree_utils.unique_trees_and_multiplicity(T_list_seed)
    print(f"N uniques trees sampled: {len(unique_seq)}")
    T_list_seed_remapped = tree_utils.remap_node_labels(T_list_seed, best_perm)
    unique_edges_list, unique_edges_count = tree_utils.get_unique_edges(T_list_seed_remapped)

    x_axis = list(range(0, len(unique_edges_list)))
    y_axis = [unique_edges_count[e].item() for e in unique_edges_list]
    if true_tree is not None:
        labels = [str(e) if e not in true_tree.edges else f'[{e[0]}, {e[1]}]' for e in unique_edges_list]
        colors = ['blue' if e not in true_tree.edges else 'orange' for e in unique_edges_list]
        for i in x_axis:
            fig = plt.scatter(x_axis[i], y_axis[i], s=30, c=colors[i])
        plt.xticks(ticks=x_axis, labels=labels, rotation=60)

    else:
        labels = [str(e) for e in unique_edges_list]
        fig = plt.plot(x_axis, y_axis, 'o')
        plt.xticks(ticks=x_axis, labels=labels, rotation=60)

    plt.ylabel('Edges count')
    plt.xlabel('Unique edges in sampled trees')
    plt.title(f'Sampled edges experiment - L: {L} K:{K} N: {N} - M: {M} - A: {A}')
    dirs = os.getcwd().split('/')
    dir_top_idx = dirs.index('victree')
    dir_path = dirs[dir_top_idx:]
    path = os.path.join(*dir_path, 'edge_probability_analysis')
    base_dir = 'output/analysis'
    test_dir_name = data_handling.create_analysis_output_catalog(path, base_dir)
    plt.savefig(test_dir_name + f"/T_edge_plot_K{K}_N{N}_M{M}_A{A}.png")
    plt.show()


def importance_weighted_trees(q_T: qT, L, true_tree=None, best_perm=None):
    """
    Given a trained qT, analyzes the edges in the trees sampled trees.
    """
    matplotlib.use('module://backend_interagg')

    # T comparisons
    N, M, K, A = (q_T.config.n_cells, q_T.config.chain_length, q_T.config.n_nodes, q_T.config.n_states)
    T_list_seed, w_T_list_seed = q_T.get_trees_sample(sample_size=L)
    unique_seq, unique_seq_idx, multiplicity = tree_utils.unique_trees_and_multiplicity(T_list_seed)
    print(f"N uniques trees sampled: {len(unique_seq)}")
    W = q_T.weight_matrix
    G = q_T.weighted_graph
    mst = nx.maximum_spanning_arborescence(G)
    pos = graphviz_layout(mst, prog="dot")
    nx.draw(mst, pos=pos)
    plt.title(f'MST plot')
    dirs = os.getcwd().split('/')
    dir_top_idx = dirs.index('victree')
    dir_path = dirs[dir_top_idx:]
    path = os.path.join(*dir_path, 'edge_probability_analysis')
    base_dir = 'output/analysis'
    test_dir_name = data_handling.create_analysis_output_catalog(path, base_dir)
    plt.savefig(test_dir_name + f"/MST.png")
    plt.show()

    max_w_tree_idx = np.argmax(w_T_list_seed)
    max_w_tree = T_list_seed[max_w_tree_idx]
    print(f"Max weight: {w_T_list_seed[max_w_tree_idx]}")
    pos = graphviz_layout(max_w_tree, prog="dot")
    nx.draw(max_w_tree, pos=pos)
    plt.title(f'Max sampled tree plot')
    dirs = os.getcwd().split('/')
    dir_top_idx = dirs.index('victree')
    dir_path = dirs[dir_top_idx:]
    path = os.path.join(*dir_path, 'edge_probability_analysis')
    base_dir = 'output/analysis'
    test_dir_name = data_handling.create_analysis_output_catalog(path, base_dir)
    plt.savefig(test_dir_name + f"/MaxWeightTree.png")
    plt.show()


if __name__ == '__main__':
    L = 100
    file_path = '../../output/P01-066/K10L5i200step0p1split/victree.model.h5'
    checkpoint_data = data_handling.read_checkpoint(file_path)
    q_T = factory_utils.construct_qT_from_checkpoint_data(checkpoint_data)
    importance_weighted_trees(q_T, L)
    edge_probability_analysis(q_T, L)