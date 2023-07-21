import random

import networkx as nx
import numpy as np
import torch
from typing import List, Tuple

from dendropy import Tree
from dendropy.calculate.treecompare import symmetric_difference
# from Espalier import MAF
# from pylabeledrf.computeLRF import *
from networkx import is_arborescence


def generate_fixed_tree(n_nodes: int, seed=0):
    return nx.random_tree(n=n_nodes, seed=seed, create_using=nx.DiGraph)


def get_unique_edges(T_list: List[nx.DiGraph], N_nodes: int = None) -> Tuple[List, torch.Tensor]:
    N_nodes = T_list[0].number_of_nodes() if N_nodes is None else N_nodes
    unique_edges_list = []
    unique_edges_count = torch.zeros(N_nodes, N_nodes, dtype=torch.int)
    for T in T_list:
        for uv in T.edges:
            if unique_edges_count[uv] == 0:
                unique_edges_count[uv] = 1
                unique_edges_list.append(uv)
            else:
                unique_edges_count[uv] += 1

    return unique_edges_list, unique_edges_count


def newick_from_eps_arr(eps_arr: np.ndarray):
    t = nx.DiGraph()
    for u, v in zip(*np.where(eps_arr > 0)):
        t.add_edge(u, v)
    return tree_to_newick(t)


def tree_to_newick(g: nx.DiGraph, root=None, weight=None):
    # make sure the graph is a tree
    assert is_arborescence(g)
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]
    subgs = []
    # sorting makes sure same trees have same newick
    for child in sorted(g[root]):
        node_str: str
        if len(g[child]) > 0:
            node_str = tree_to_newick(g, root=child, weight=weight)
        else:
            node_str = str(child)

        if weight is not None:
            node_str += ':' + str(g.get_edge_data(root, child)[weight])
        subgs.append(node_str)
    return "(" + ','.join(subgs) + ")" + str(root)


def top_k_trees_from_sample(t_list, w_list, k: int = 0, by_weight=True, nx_graph=False):
    """
    Parameters
    ----------
    t_list list of sampled trees
    w_list sampled trees weights
    k number of unique top trees in output, if 0, return all unique trees
    by_weight bool, if true sorts by decreasing sum of weights.
        if false, sorts by decreasing number of trees (cardinality)
    nx_graph bool, if true nx.DiGraph object is returned, if false newick str is returned

    Returns
    -------
    list of tuples, top k trees depending on chosen order
    """
    unique_trees = {}
    for t, w in zip(t_list, w_list):
        t_newick: str = tree_to_newick(t)
        if t_newick not in unique_trees:
            unique_trees[t_newick] = {
                'nx-tree': t,
                'weight': 0.,
                'count': 0
            }
        diff = nx.difference(unique_trees[t_newick]['nx-tree'], t).size()
        assert diff == 0, \
            f"same string but different sets of edges: {t_newick} -> {[e for e in unique_trees[t_newick]['nx-tree'].edges]}," \
            f" {[e for e in t.edges]} | diff = {diff}"
        unique_trees[t_newick]['weight'] += w
        unique_trees[t_newick]['count'] += 1

        for alt_t in unique_trees:
            if alt_t != t_newick:
                # check effectively that all trees with different newick have
                # different sets of edges
                assert nx.difference(unique_trees[alt_t]['nx-tree'], t).size() > 0

    if k == 0:
        k = len(unique_trees)

    sorted_trees: [(str | nx.DiGraph, float)]
    sorted_trees = [(t_dat['nx-tree'] if nx_graph else t_str,
                     t_dat['weight'] if by_weight else t_dat['count'])
                    for t_str, t_dat in sorted(unique_trees.items(), key=lambda x: x[1]['weight'], reverse=True)]
    return sorted_trees[:k]


def networkx_tree_to_dendropy(T: nx.DiGraph, root) -> Tree:
    return Tree.get(data=tree_to_newick(T, root) + ";", schema="newick")


def calculate_SPR_distance(T_1: nx.DiGraph, T_2: nx.DiGraph):
    raise NotImplementedError("SPR distance not well defined for labeled trees.")
    T_1 = networkx_tree_to_dendropy(T_1, 0)
    T_2 = networkx_tree_to_dendropy(T_2, 0)
    spr_dist = None
    # spr_dist = MAF.get_spr_dist(T_1, T_2)
    return spr_dist


def calculate_Robinson_Foulds_distance(T_1: nx.DiGraph, T_2: nx.DiGraph):
    raise NotImplementedError("RF distance not well defined for labeled trees (only leaf labeled trees).")
    T_1 = networkx_tree_to_dendropy(T_1, 0)
    T_2 = networkx_tree_to_dendropy(T_2, 0)
    return symmetric_difference(T_1, T_2)


def calculate_Labeled_Robinson_Foulds_distance(T_1: nx.DiGraph, T_2: nx.DiGraph):
    "Package from: https://github.com/DessimozLab/pylabeledrf"
    T_1 = networkx_tree_to_dendropy(T_1, 0)
    # t1 = parseEnsemblLabels(T_1)
    T_2 = networkx_tree_to_dendropy(T_2, 0)
    # t2 = parseEnsemblLabels(T_2)
    # computeLRF(t1, t2)
    return symmetric_difference(T_1, T_2)


def calculate_graph_distance(T_1: nx.DiGraph, T_2: nx.DiGraph, roots=(0, 0), labeled_distance=True):
    if labeled_distance:
        # edge_match = lambda (u1,v1), (u2,v2) : u1==u2 and v1
        node_match = lambda u1, u2: u1 == u2
    distance = nx.graph_edit_distance(T_1, T_2, roots=roots, node_match=node_match)
    return distance


def relabel_nodes(T, labeling):
    mapping = {}
    orig = list(range(0, 1 + np.max(labeling)))
    for (old, new) in zip(orig, labeling):
        mapping[old] = new
    return nx.relabel_nodes(T, mapping, copy=True)


def relabel_trees(T_list: list[nx.DiGraph], labeling):
    return [relabel_nodes(T, labeling) for T in T_list]


def distances_to_true_tree(true_tree, trees_to_compare: list[nx.DiGraph], labeling=None):
    L = len(trees_to_compare)
    distances = np.zeros(L, )
    for l, T in enumerate(trees_to_compare):
        if labeling is not None:
            T = relabel_nodes(T, labeling)
        distances[l] = calculate_graph_distance(true_tree, T)
    return distances


def to_prufer_sequences(T_list: list[nx.DiGraph]):
    return [nx.to_prufer_sequence(T) for T in T_list]


def unique_trees(prufer_list: list[list[int]]):
    unique_seq = []
    unique_seq_idx = []
    for (i, seq) in enumerate(prufer_list):
        if seq in unique_seq:
            continue
        else:
            unique_seq.append(seq)
            unique_seq_idx.append(i)
    return unique_seq, unique_seq_idx


def unique_trees_and_multiplicity(T_list_or_prufer_seq_list):
    if type(T_list_or_prufer_seq_list[0]) == nx.DiGraph:
        undir_trees = to_undirected(T_list_or_prufer_seq_list)
        prufer_seq_list = to_prufer_sequences(undir_trees)
    else:
        prufer_seq_list = T_list_or_prufer_seq_list

    unique_seq = []
    unique_seq_idx = []
    multiplicity = []
    for (i, seq) in enumerate(prufer_seq_list):
        if seq in unique_seq:
            idx = unique_seq.index(seq)
            multiplicity[idx] += 1
        else:
            unique_seq.append(seq)
            unique_seq_idx.append(i)
            multiplicity.append(1)
    return unique_seq, unique_seq_idx, multiplicity


def to_undirected(T_list: list[nx.DiGraph]):
    return [nx.to_undirected(T) for T in T_list]


def get_two_step_connections(T: nx.DiGraph, u):
    two_step_neighbours = set()
    for v in nx.neighbors(T, u):
        two_step_neighbours.update(set(nx.neighbors(T, v)))

    return list(two_step_neighbours)


def get_all_two_step_connections(T: nx.DiGraph):
    node_two_step_neighbours_dict = {}
    for u in T.nodes:
        u_two_order_neighbours = get_two_step_connections(T, u)
        node_two_step_neighbours_dict[u] = u_two_order_neighbours

    return node_two_step_neighbours_dict


def remap_edge_labels(T_list, perm):
    K = len(perm)
    perm_dict = {i: perm[i] for i in range(K)}
    T_list_remapped = []
    for T in T_list:
        T_remapped = nx.relabel_nodes(T, perm_dict, copy=True)
        T_list_remapped.append(T_remapped)

    return T_list_remapped


def generate_all_directed_unlabeled_tree_topologies(n):
    def generate_trees(nodes):
        if len(nodes) == 1:
            return [nodes]

        all_trees = []
        for i in range(1, len(nodes) + 1):  # Number of children for the root node
            root = nodes[0]
            children_combinations = generate_trees(nodes[1:])

            for combination in children_combinations:
                tree = [root] + combination[:i]
                if i < len(combination):
                    tree.append(combination[i:])
                all_trees.append(tree)

        return all_trees

    # Generate all permutations of nodes
    nodes = list(range(1, n + 1))
    all_permutations = generate_trees(nodes)

    return all_permutations


def generate_directed_tree_from_pruefer(pruefer_sequence):
    n = len(pruefer_sequence) + 2
    freq = [0] * (n + 1)

    # Calculate frequency of each node
    for node in pruefer_sequence:
        freq[node] += 1

    tree = []

    for node in pruefer_sequence:
        # Find the smallest unused node (with frequency 0)
        for i in range(1, n + 1):
            if freq[i] == 0:
                tree.append((node, i))
                freq[node] -= 1
                freq[i] += 1
                break

    # Find the remaining two nodes with frequency 0 and connect them
    last_nodes = [i for i in range(1, n + 1) if freq[i] == 0]
    tree.append((last_nodes[0], last_nodes[1]))

    # Choose a root node and direct all the edges away from the root
    root = min(last_nodes)
    directed_tree = [(parent, child) if parent != root else (child, parent) for parent, child in tree]

    return directed_tree


def generate_similar_tree(base_tree):
    # Create a copy of the base tree
    similar_tree = nx.Graph(base_tree)

    # Get a list of all edges in the tree
    edges = list(similar_tree.edges())

    # Randomly shuffle the edges
    random.shuffle(edges)

    # Reconstruct the tree with the shuffled edges
    similar_tree = nx.Graph()
    similar_tree.add_nodes_from(base_tree.nodes())
    for u, v in edges:
        if nx.is_tree(similar_tree):
            similar_tree.add_edge(u, v)

    return similar_tree


def perform_SPR_move(tree: nx.DiGraph):
    def select_random_internal_node(exclude_nodes, tree: nx.DiGraph):
        internal_nodes = [u for u in tree.nodes if u not in exclude_nodes or not nx.descendants(tree, u) == set()]
        return random.choice(internal_nodes)

    def remove_subtree(parent_node):
        nw_tree = tree_to_newick(tree, 0)
        removed_subtree = tree.remove_node(parent_node) 
        tree[parent_node] = [tree[parent_node][0]]  # Keep only the parent node
        return removed_subtree

    def reinsert_subtree(parent_edge, reinsertion_point):
        tree[reinsertion_point].extend(parent_edge)

    if len(tree.nodes) <= 2:
        return tree  # No SPR moves can be performed on trees with two or fewer nodes

    parent_node = select_random_internal_node(exclude_nodes=[0], tree=tree)
    removed_subtree = remove_subtree(parent_node)

    parent_edge = random.choice(removed_subtree)
    reinsertion_point = select_random_internal_node(exclude_nodes=[0, parent_node])
    reinsert_subtree(parent_edge, reinsertion_point)

    return tree