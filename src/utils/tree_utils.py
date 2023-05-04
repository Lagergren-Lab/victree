import dendropy
import networkx as nx
import numpy as np
import torch
from typing import List, Tuple

from Espalier import MAF
from pylabeledrf.computeLRF import *
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

def forward_messages_markov_chain(initial_probs: torch.Tensor, transition_probabilities: torch.Tensor):
    chain_length = transition_probabilities.shape[0] + 1
    n_states = initial_probs.shape[0]
    alpha = torch.zeros((chain_length, n_states))  # Forward recursion variable
    alpha[0] = initial_probs

    for n in range(1, chain_length):
        alpha[n] = torch.einsum("ij, i -> j", transition_probabilities[n - 1], alpha[n - 1])
    return alpha


def backward_messages_markov_chain(transition_probabilities: torch.Tensor):
    # alpha_m = sum_{n-1}
    M, n_states, _ = transition_probabilities.shape
    beta = torch.zeros((M, n_states))  # Forward recursion variable

    # backward
    beta[M - 1] = 1.
    for rm in range(1, M):
        beta[M - rm - 1] = torch.einsum("j, ij -> i", beta[M - rm], transition_probabilities[M - rm - 1])
    return beta


def two_slice_marginals_markov_chain_given_alpha_beta(alpha: torch.Tensor, transition_probabilities: torch.Tensor,
                                                      beta: torch.Tensor) -> torch.Tensor:
    M, n_states = alpha.shape
    two_slice_marginals_tensor = torch.zeros(transition_probabilities.shape)
    for m in range(M - 1):
        unnormalized_two_slice_marginals = torch.einsum("i, ij, j -> ij", alpha[m], transition_probabilities[m],
                                                        beta[m])
        two_slice_marginals_tensor[m] = unnormalized_two_slice_marginals / torch.sum(unnormalized_two_slice_marginals)
    return two_slice_marginals_tensor


def two_slice_marginals_markov_chain(initial_state: torch.Tensor, transition_probabilities: torch.Tensor):
    """
    :param N: Chain length
    :param initial_state: markov model initial state probability tensor                 - (M x 1)
    :param transition_probabilities: markov model probability tensor                    - (N x M x M)
    :return: pairwise probability tensor [p(X_1, X_2), p(X_2, X_3) ... p(X_{N-1}, X_N)]    - (N-1 x M)
    """
    alpha = forward_messages_markov_chain(initial_state, transition_probabilities)
    beta = backward_messages_markov_chain(transition_probabilities)

    return two_slice_marginals_markov_chain_given_alpha_beta(alpha, transition_probabilities, beta)


def one_slice_marginals_markov_chain(initial_state: torch.Tensor, transition_probabilities: torch.Tensor):
    return forward_messages_markov_chain(initial_state, transition_probabilities)


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


def top_k_trees_from_sample(t_list, w_list, k, by_weight=True, nx_graph=False):
    """
    Parameters
    ----------
    t_list list of sampled trees
    w_list sampled trees weights
    k number of unique top trees in output
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

    sorted_trees: [(str, float)]
    sorted_trees = [(t_dat['nx-tree'] if nx_graph else t_str,
                     t_dat['weight'] if by_weight else t_dat['count'])
                    for t_str, t_dat in sorted(unique_trees.items(), key=lambda x: x[1]['weight'], reverse=True)]
    return sorted_trees[:k]


def networkx_tree_to_dendropy(T: nx.DiGraph, root) -> dendropy.Tree:
    return dendropy.Tree.get(data=tree_to_newick(T, root) + ";", schema="newick")


def calculate_SPR_distance(T_1: nx.DiGraph, T_2: nx.DiGraph):
    raise NotImplementedError("SPR distance not well defined for labeled trees.")
    T_1 = networkx_tree_to_dendropy(T_1, 0)
    T_2 = networkx_tree_to_dendropy(T_2, 0)
    return MAF.get_spr_dist(T_1, T_2)


def calculate_Robinson_Foulds_distance(T_1: nx.DiGraph, T_2: nx.DiGraph):
    raise NotImplementedError("RF distance not well defined for labeled trees (only leaf labeled trees).")
    T_1 = networkx_tree_to_dendropy(T_1, 0)
    T_2 = networkx_tree_to_dendropy(T_2, 0)
    return dendropy.calculate.treecompare.symmetric_difference(T_1, T_2)


def calculate_Labeled_Robinson_Foulds_distance(T_1: nx.DiGraph, T_2: nx.DiGraph):
    "Package from: https://github.com/DessimozLab/pylabeledrf"
    T_1 = networkx_tree_to_dendropy(T_1, 0)
    t1 = parseEnsemblLabels(T_1)
    T_2 = networkx_tree_to_dendropy(T_2, 0)
    t2 = parseEnsemblLabels(T_2)
    computeLRF(t1, t2)
    return dendropy.calculate.treecompare.symmetric_difference(T_1, T_2)


def calculate_graph_distance(T_1: nx.DiGraph, T_2: nx.DiGraph, roots=(0, 0), labeled_distance=True):
    if labeled_distance:
        #edge_match = lambda (u1,v1), (u2,v2) : u1==u2 and v1
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
    distances = np.zeros(L,)
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


def unique_trees_and_multiplicity(prufer_list: list[list[int]]):
    unique_seq = []
    unique_seq_idx = []
    multiplicity = []
    for (i, seq) in enumerate(prufer_list):
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


def get_unique_trees_and_multiplicity(T_list: list[nx.DiGraph]):
    T_list_undir = to_undirected(T_list)
    prufer_seqs_list = to_prufer_sequences(T_list_undir)
    unique_seq, unique_seq_idx, multiplicity = unique_trees_and_multiplicity(prufer_seqs_list)
    return [T_list[i] for i in unique_seq_idx], unique_seq_idx, multiplicity


def get_all_prufer_seq(K):
    seq = list(range(0, K-2))
    return itertools.permutations(seq)


def get_all_topologies(K):
    raise NotImplementedError
    T_list = []
    prufer_seq = get_all_prufer_seq(K)
    for pruf in prufer_seq:
        T_list.append(nx.from_prufer_sequence(pruf))
    return T_list