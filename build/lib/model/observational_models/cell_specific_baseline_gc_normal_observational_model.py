import torch


class CellSpecificBaselineGCNormalObsModel:

    def __init__(self, N, M):
        self.N = N
        self.M = M

    def simulate_data(self, c, z, nu0, lambda0, alpha0, beta0):
        N, M = (self.N, self.M)
        tau = torch.distributions.Gamma(alpha0, beta0).sample((N,))
        mu = torch.distributions.Normal(nu0, 1. / torch.sqrt(lambda0 * tau)).sample()
        obs_mean = c[z, :] * mu[:, None]  # n_cells x chain_length
        scale_expanded = torch.pow(tau, -1 / 2).reshape(-1, 1).expand(-1, M)
        y = torch.distributions.Normal(obs_mean, scale_expanded).sample()
        y = y.T
        assert y.shape == (M, N)
        return y, mu, tau
