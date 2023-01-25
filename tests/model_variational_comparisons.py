from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as f
import unittest

from variational_distributions.var_dists import qC, qZ, qPi, qMuTau, qEpsilonMulti
from inference.copy_tree import VarDistFixedTree
from model.generative_model import GenerativeModel


def compare_qC_and_true_C(true_C, q_c: qC, threshold):
    marginals = q_c.single_filtering_probs
    max_prob_cat = torch.argmax(marginals, dim=-1)
    n_diff = torch.sum(true_C != max_prob_cat)
    assert n_diff <= threshold


def compare_qZ_and_true_Z(true_Z, q_z: qZ):
    N, K = q_z.pi.shape
    z_true_one_hot = f.one_hot(true_Z.long(), num_classes=K)

    total_misclassifications = torch.sum(true_Z.float() != torch.argmax(q_z.pi))
    kl_distance = torch.kl_div(q_z.pi, z_true_one_hot)


def compare_qMuTau_with_true_mu_and_tau(true_mu, true_tau, q_mt):
    N = true_mu.shape
    square_dist = torch.pow(true_mu - q_mt.nu, 2)
    total_square_dist = square_dist.sum()
    print(f"total square dist mu: {total_square_dist}")

    square_dist_tau = torch.pow(true_tau - q_mt.exp_tau(), 2)
    total_square_dist_tau = square_dist_tau.sum()
    print(f"total square dist tau: {total_square_dist_tau}")

def fixed_T_comparisons(obs, true_C, true_Z, true_pi, true_mu, true_tau, true_epsilon, q_c: qC, q_z: qZ, qpi: qPi, q_mt: qMuTau):
    n_diff = compare_qC_and_true_C(true_C, q_c, threshold=10)
    compare_qZ_and_true_Z(true_Z, q_z)
    compare_qMuTau_with_true_mu_and_tau(true_mu, true_tau, q_mt)

