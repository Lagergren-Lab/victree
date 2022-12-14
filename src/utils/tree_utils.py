import networkx as nx
import torch
from typing import List, Tuple


def get_unique_edges(T_list: List[nx.DiGraph], N_nodes: int) -> Tuple[List, torch.Tensor]:
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


def forward_messages_markov_chain(initial_probs: torch.Tensor, transition_probabilities: torch.Tensor, N: int):
    A = initial_probs.shape[0]
    alpha = torch.zeros((N, A))  # Forward recursion variable
    # TODO: handle update with full eta1 input
    alpha[0] = initial_probs

    for n in range(1, N):
        alpha[n] = torch.einsum("ij, i -> j", transition_probabilities[n - 1], alpha[n - 1])
    return alpha


def backward_messages_markov_chain(transition_probabilities: torch.Tensor, N: int):
    # alpha_m = sum_{n-1}
    A, M, M = transition_probabilities.shape
    beta = torch.zeros((N, M))  # Forward recursion variable

    # backward
    beta[N - 1] = 1
    for rn in range(1, N):
        beta[N - rn - 1] = torch.einsum("j, ij -> i", beta[N - rn], transition_probabilities[N - rn - 1])
    return beta


def two_slice_marginals_markov_chain_given_alpha_beta(alpha: torch.Tensor, transition_probabilities: torch.Tensor,
                                     beta: torch.Tensor) -> torch.Tensor:
    N, M = alpha.shape
    two_slice_marginals_tensor = torch.zeros((N - 1, M, M))
    for n in range(0, N - 1):
        unnormalized_two_slice_marginals = torch.einsum("i, ij, j -> ij", alpha[n], transition_probabilities[n],
                                                        beta[n])
        two_slice_marginals_tensor[n] = unnormalized_two_slice_marginals / torch.sum(unnormalized_two_slice_marginals)
    return two_slice_marginals_tensor


def two_slice_marginals_markov_chain(initial_state: torch.Tensor, transition_probabilities: torch.Tensor, N: int):
    """
    :param N: Chain length
    :param initial_state: markov model initial state probability tensor                 - (M x 1)
    :param transition_probabilities: markov model probability tensor                    - (N x M x M)
    :return: pairwise probability tensor [p(X_1, X_2), p(X_2, X_3) ... p(X_{N-1}, X_N)]    - (N-1 x M)
    """
    alpha = forward_messages_markov_chain(initial_state, transition_probabilities, N)
    beta = backward_messages_markov_chain(transition_probabilities, N)

    return two_slice_marginals_markov_chain_given_alpha_beta(alpha, transition_probabilities, beta)


def one_slice_marginals_markov_chain(initial_state: torch.Tensor, transition_probabilities: torch.Tensor, N: int):
    return forward_messages_markov_chain(initial_state, transition_probabilities, N)


if __name__ == '__main__':
    N_states = 2
    init_state = torch.ones(N_states) * 1.0 / N_states
    N_stages = 5
    transitions = torch.ones((N_stages, N_states, N_states)) * 1.0 / N_states
    pairwise_joint = two_slice_marginals_markov_chain(init_state, transitions, N_stages)
    print(pairwise_joint)
