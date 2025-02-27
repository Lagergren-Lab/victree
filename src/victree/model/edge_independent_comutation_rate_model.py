import torch


class EdgeIndependentComutationProbModel:
    """
    Generative model for edge specific comutation transition probabilites: p(eps | T)
    """
    def __init__(self, K):
        self.K = K

    def simulate_data(self, T, a, b, eps0_a, eps0_b):
        eps = {}
        for u, v in T.edges:
            eps[u, v] = torch.distributions.Beta(a, b).sample()
            T.edges[u, v]['weight'] = eps[u, v]
        eps0 = torch.distributions.Beta(eps0_a, eps0_b).sample()
        return eps, eps0