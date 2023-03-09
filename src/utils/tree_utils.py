import networkx as nx
import torch
from typing import List, Tuple

from networkx import is_arborescence


def generate_fixed_tree(n_nodes: int):
    return nx.random_tree(n=n_nodes, seed=0, create_using=nx.DiGraph)


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


if __name__ == '__main__':
    N_states = 2
    init_state = torch.ones(N_states) * 1.0 / N_states
    N_stages = 5
    transitions = torch.ones((N_stages, N_states, N_states)) * 1.0 / N_states
    pairwise_joint = two_slice_marginals_markov_chain(init_state, transitions)
    print(pairwise_joint)


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
