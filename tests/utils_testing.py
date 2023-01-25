from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as f
from pyro import poutine

import simul
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


def get_tree_K_nodes_random(K):
    T = nx.DiGraph()
    T.add_edge(0, 1)
    nodes_in_T = [0, 1]
    nodes_not_in_T = list(range(2, K))
    for i in range(len(nodes_not_in_T)):
        parent = np.random.choice(nodes_in_T)
        child = np.random.choice(nodes_not_in_T)
        T.add_edge(parent, child)
        nodes_in_T.append(child)
        nodes_not_in_T.remove(child)

    return T


def get_random_q_C(M, A):
    q_C_init = torch.rand(A)
    q_C_transitions_unnormalized = torch.rand((M-1, A, A))
    q_C_transitions = f.normalize(q_C_transitions_unnormalized, p=1, dim=2)
    return q_C_init, q_C_transitions


def get_root_q_C(M, A):
    q_C_init = torch.zeros(A)
    q_C_init[2] = 1
    q_C_transitions = torch.zeros((M-1, A, A))
    diag_A = torch.ones(A)
    for m in range(M-1):
        q_C_transitions[m] = torch.diag(diag_A, 0)
    return q_C_init, q_C_transitions


def get_two_simple_trees_with_random_qCs(M, A) -> Tuple[List[nx.DiGraph], torch.Tensor]:
    T_1 = get_tree_three_nodes_balanced()
    T_2 = get_tree_three_nodes_chain()
    T_list = [T_1, T_2]
    N = 3
    q_C_0_init, q_C_0_transitions = get_root_q_C(M, A)
    q_C_0_pairwise_marginals = tree_utils.two_slice_marginals_markov_chain(q_C_0_init, q_C_0_transitions)
    q_C_1_init, q_C_1_transitions = get_random_q_C(M, A)
    q_C_1_pairwise_marginals = tree_utils.two_slice_marginals_markov_chain(q_C_1_init, q_C_1_transitions)
    q_C_2_init, q_C_2_transitions = get_random_q_C(M, A)
    q_C_2_pairwise_marginals = tree_utils.two_slice_marginals_markov_chain(q_C_2_init, q_C_2_transitions)
    q_C_pairwise_marginals = torch.zeros(N, M - 1, A, A)
    q_C_pairwise_marginals[0] = q_C_0_pairwise_marginals
    q_C_pairwise_marginals[1] = q_C_1_pairwise_marginals
    q_C_pairwise_marginals[2] = q_C_2_pairwise_marginals
    return T_list, q_C_pairwise_marginals


def simul_data_pyro_full_model(data, n_cells, n_sites, n_copy_states, tree: nx.DiGraph,
                               mu_0=torch.tensor(1.),
                               lambda_0=torch.tensor(1.),
                               alpha0=torch.tensor(1.),
                               beta0=torch.tensor(1.),
                               a0=torch.tensor(1.0),
                               b0=torch.tensor(10.0),
                               dir_alpha0=torch.tensor(1.0)
                               ):
    model_tree_markov_full = simul.model_tree_markov_full
    unconditioned_model = poutine.uncondition(model_tree_markov_full)
    C, y, z, pi, mu, tau, eps = unconditioned_model(data, n_cells, n_sites, n_copy_states, tree,
                                                    mu_0,
                                                    lambda_0,
                                                    alpha0,
                                                    beta0,
                                                    a0,
                                                    b0,
                                                    dir_alpha0)

    return C, y, z, pi, mu, tau, eps
