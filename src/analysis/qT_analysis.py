from matplotlib import pyplot as plt

from inference.victree import VICTree
from utils import tree_utils, data_handling


def edge_probability_analysis(victree: VICTree, true_tree=None, best_perm=None):
    """
    Given a trained qT, analyzes the edges in the trees sampled trees.
    """

    # T comparisons
    T_list_seed, w_T_list_seed = victree.q.t.get_trees_sample(sample_size=100)
    unique_seq, unique_seq_idx, multiplicity = tree_utils.unique_trees_and_multiplicity(T_list_seed)
    print(f"N uniques trees sampled: {len(unique_seq)}")
    T_list_seed_remapped = tree_utils.remap_edge_labels(T_list_seed, best_perm)
    unique_edges_list, unique_edges_count = tree_utils.get_unique_edges(T_list_seed_remapped)
    x_axis = list(range(0, len(unique_edges_list)))
    y_axis = [unique_edges_count[e].item() for e in unique_edges_list]
    true_tree_edges = [unique_edges_count[e].item() for e in true_tree.edges]
    labels = [str(e) if e not in true_tree.edges else f'[{e[0]}, {e[1]}]' for e in unique_edges_list]
    colors = ['blue' if e not in true_tree.edges else 'orange' for e in unique_edges_list]
    for i in x_axis:
        plt.scatter(x_axis[i], y_axis[i], s=30, c=colors[i])
    #plt.plot(x_axis, true_tree_edges, 'x')
    plt.xticks(ticks=x_axis, labels=labels, rotation=60)
    plt.ylabel('Edges count')
    plt.xlabel('Unique edges in sampled trees')
    plt.title(f'Sampled edges experiment seed {seed} - L: {100} K:{K} N: {N} - M: {M} - A: {A}')
    if save_plot:
        dirs = os.getcwd().split('/')
        dir_top_idx = dirs.index('experiments')
        dir_path = dirs[dir_top_idx:]
        path = os.path.join(*dir_path, self.__class__.__name__, sys._getframe().f_code.co_name)
        base_dir = '../../test_output'
        test_dir_name = tests.utils_testing.create_experiment_output_catalog(path, base_dir)
        plt.savefig(test_dir_name + f"/T_edge_plot_seed{seed}_K{K}_N{N}_M{M}_A{A}.png")
        plt.close()


if __name__ == '__main__':

    victree = data_handling.load_victree_model()