import torch


def log_beta_function(x: torch.Tensor)-> torch.float:
    x_0 = torch.sum(x)
    return torch.sum(torch.lgamma(x)) / torch.lgamma(x_0)


def cayleys_formula(K):
    return K**(K-2)