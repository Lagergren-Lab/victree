import os

from matplotlib import pyplot as plt

from inference.victree import VICTree
from utils import tree_utils, factory_utils, data_handling
from utils.config import Config
from variational_distributions.var_dists import qT


def edge_probability_analysis(q_T: qT, L, true_tree=None, best_perm=None):
    """
    Given a trained qT, analyzes the edges in the trees sampled trees.
    """

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
        fig = plt.plot(x_axis, y_axis, 'o')
        plt.xticks(ticks=x_axis, rotation=60)

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
    plt.close()


if __name__ == '__main__':
    L = 10
    file_path = '../../output/checkpoint_k6a7n1105m6206.h5'
    checkpoint_data = data_handling.read_checkpoint(file_path)
    q_T = factory_utils.construct_qT_from_checkpoint_data(checkpoint_data)
    edge_probability_analysis(q_T, L)