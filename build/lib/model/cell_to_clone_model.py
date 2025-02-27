import logging

import torch


class CellToCloneModel:

    def __init__(self, N, K):
        self.N = N
        self.K = K

    def simulate_data(self, dir_delta):
        N, K = (self.N, self.K)
        logging.debug(f'Component assignments simulation - delta: {dir_delta}')
        if isinstance(dir_delta, float):
            dir_alpha_tensor = torch.ones(K) * dir_delta
        else:
            dir_alpha_tensor = torch.tensor(dir_delta)
        pi = torch.distributions.Dirichlet(dir_alpha_tensor).sample()
        z = torch.distributions.Categorical(pi).sample((N,))
        return z, pi