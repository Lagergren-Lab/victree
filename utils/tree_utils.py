import networkx as nx
import torch


def get_unique_edges(T_list: nx.Graph):
    return None


def forward_backward_markov_model(markov_model: torch.Tensor, transition_probabilities: torch.Tensor):
    """
    :param markov_model: markov model probability tensor (N x M)
    :return: joint probability tensor (N-1 x M)
    """
    N, M = markov_model.size()
    pairwise_marginals = torch.zeros((N - 1, M, M))
    alpha_m = torch.zeros((N - 1, M, M))    # Forward recursion variable
    beta_m = torch.zeros((N - 1, M))        # Backward recursion variable

    # Initial states
    alpha_m[0] = torch.multiply(transition_probabilities, markov_model[0])
    beta_m[N - 2] = torch.ones(M)

    # TODO: Replace loops with extended einsum operations
    # forward
    for n in range(1, N-1):
        alpha_m[n] = torch.einsum("ij, jk -> ik", transition_probabilities, alpha_m[n - 1])

    # backward
    for n in reversed(range(0, N-2)):
        beta_m[n] = torch.einsum("li, l -> i", transition_probabilities, beta_m[n + 1])

    # pairwise marginals
    for n in range(0, N-1):
        pairwise_marginals[n] = torch.multiply(alpha_m[n], beta_m[n])

    return pairwise_marginals


if __name__ == '__main__':
    mm = torch.tensor([[1., 0.0], [0.5, 0.5], [0.3, 0.7]])
    t = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    pairwise_joint = forward_backward_markov_model(mm, t)
    print(pairwise_joint)
