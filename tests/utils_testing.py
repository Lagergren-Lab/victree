from typing import List, Tuple

import networkx as nx
import torch
import torch.nn.functional as f

from utils import tree_utils


def get_tree_three_nodes_balanced():
    T_1 = nx.DiGraph()
    T_1.add_edge(0, 1)
    T_1.add_edge(0, 2)
    return T_1


def get_tree_three_nodes_chain():
    T_2 = nx.DiGraph()
    T_2.add_edge(0, 1)
    T_2.add_edge(1, 2)
    return T_2


def get_random_q_C(M, A):
    q_C_init = torch.rand(A)
    q_C_transitions_unnormalized = torch.rand((M, A, A))
    q_C_transitions = f.normalize(q_C_transitions_unnormalized, p=1, dim=2)
    return q_C_init, q_C_transitions


def get_root_q_C(M, A):
    q_C_init = torch.zeros(A)
    q_C_init[2] = 1
    q_C_transitions = torch.zeros((M, A, A))
    diag_A = torch.ones(A)
    for m in range(M):
        q_C_transitions[m] = torch.diag(diag_A, 0)
    return q_C_init, q_C_transitions


def get_two_simple_trees_with_random_qCs(M, A) -> Tuple[List[nx.DiGraph], torch.Tensor]:
    T_1 = get_tree_three_nodes_balanced()
    T_2 = get_tree_three_nodes_chain()
    T_list = [T_1, T_2]
    N = 3
    q_C_0_init, q_C_0_transitions = get_root_q_C(M, A)
    q_C_0_pairwise_marginals = tree_utils.two_slice_marginals_markov_chain(q_C_0_init, q_C_0_transitions, M)
    q_C_1_init, q_C_1_transitions = get_random_q_C(M, A)
    q_C_1_pairwise_marginals = tree_utils.two_slice_marginals_markov_chain(q_C_1_init, q_C_1_transitions, M)
    q_C_2_init, q_C_2_transitions = get_random_q_C(M, A)
    q_C_2_pairwise_marginals = tree_utils.two_slice_marginals_markov_chain(q_C_2_init, q_C_2_transitions, M)
    q_C_pairwise_marginals = torch.zeros(N, M - 1, A, A)
    q_C_pairwise_marginals[0] = q_C_0_pairwise_marginals
    q_C_pairwise_marginals[1] = q_C_1_pairwise_marginals
    q_C_pairwise_marginals[2] = q_C_2_pairwise_marginals
    return T_list, q_C_pairwise_marginals
