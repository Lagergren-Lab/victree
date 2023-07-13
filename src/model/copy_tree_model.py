import logging

import networkx as nx
import torch

from utils import eps_utils


class CopyTreeModel:

    def __init__(self, M, K, A):
        self.M = M
        self.K = K
        self.A = A

    def simulate_data(self, T, eps0, eps):
        K, M, A = (self.K, self.M, self.A)
        logging.debug(f'Copy Tree data simulation - eps0: {eps0}, eps: {eps}')

        # generate copy numbers
        c = torch.empty((K, M), dtype=torch.long)
        c[0, :] = 2 * torch.ones(M)
        h_eps0_cached = eps_utils.h_eps0(A, eps0)
        for u, v in nx.bfs_edges(T, source=0):
            t0 = h_eps0_cached[c[u, 0], :]
            c[v, 0] = torch.distributions.Categorical(probs=t0).sample()
            h_eps_uv = eps_utils.h_eps(A, eps[u, v])
            for m in range(1, M):
                # j', j, i', i
                transition = h_eps_uv[:, c[v, m - 1], c[u, m], c[u, m - 1]]
                c[v, m] = torch.distributions.Categorical(probs=transition).sample()

        return c