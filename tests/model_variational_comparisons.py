import itertools

import torch
import torch.distributions as dist
import torch.nn.functional as f
from sklearn.metrics import adjusted_rand_score

from variational_distributions.var_dists import qC, qZ, qPi, qMuTau


def compare_qC_and_true_C(true_C, q_c: qC, threshold=10):
    """
    Compares the argmax of the qC categorical distribution with the true C used to generate the data.
    :param true_C:
    :param q_c:
    :param threshold:
    :return:
    """
    K, M = true_C.shape
    marginals = q_c.single_filtering_probs
    perms_non_root = list(itertools.permutations(range(1, K)))
    perms = []
    for perm in perms_non_root:
        perms.append([0] + list(perm))

    max_prob_cat = torch.argmax(marginals, dim=-1)  # Account for label switching
    if K < 8:  # memory issues for larger K
        n_diff = torch.min((max_prob_cat[perms, :] != true_C).sum(2).sum(1))
    else:
        print(f"Start q(C) evaluation w.r.t. label switching on {len(perms)} permutations")
        n_diff = K * M
        for perm in perms:
            n_diff_i = (max_prob_cat[perm, :] != true_C).sum()
            if n_diff_i < n_diff:
                n_diff = n_diff_i
    print(f"Number of different true C and argmax(q(C)): {n_diff} out of {K*M} states")
    #assert n_diff <= threshold, f"Number of different true C and argmax(q(C)): {n_diff}"


def compare_qZ_and_true_Z(true_Z, q_z: qZ):
    """
    Compares the argmax of the parameters of qZ, i.e., the categorical probabilites, with the true cell-to-clone assignments used to generate the data.
    :param true_Z: True cell-to-clone assignments.
    :param q_z: qZ variational distribtion object.
    :return:
    """
    N, K = q_z.pi.shape
    perms = list(itertools.permutations(range(K)))
    total_misclassifications = torch.sum(true_Z.float() != torch.argmax(q_z.pi, dim=1))
    max_prob_qZ = torch.argmax(q_z.pi, dim=1)
    ari = adjusted_rand_score(max_prob_qZ, true_Z)
    print(f"Adjust rand score between q(Z) and true Z: {ari}")


def compare_qMuTau_with_true_mu_and_tau(true_mu, true_tau, q_mt):
    N = true_mu.shape
    square_dist_mu = torch.pow(true_mu - q_mt.nu, 2)
    print(f"Mean square dist mu: {square_dist_mu.mean()} +- ({square_dist_mu.std()})")
    print(f"Max square dist mu: {square_dist_mu.max()}")

    square_dist_tau = torch.pow(true_tau - q_mt.exp_tau(), 2)
    print(f"Mean square dist tau: {square_dist_tau.mean()} +- ({square_dist_tau.std()})")
    print(f"Max square dist tau: {square_dist_tau.max()}")


def compare_obs_likelihood_under_true_vs_var_model(obs, true_C, true_Z, true_mu, true_tau, q_c, q_z, q_mt):
    """
    Evaluates the likelihood of the data, obs, using the variables and parameters of the true model used to generate
    the data with the variables and parameters of expectation values of the variational distributions.
    :param obs:
    :param true_C:
    :param true_Z:
    :param true_mu:
    :param true_tau:
    :param q_c:
    :param q_z:
    :param q_mt:
    :return:
    """
    qC_marginals = q_c.single_filtering_probs
    K, M, A = qC_marginals.shape
    N = true_mu.shape

    max_prob_cat = torch.argmax(qC_marginals, dim=-1)
    exp_var_mu = q_mt.nu
    exp_var_tau = q_mt.exp_tau() if type(q_mt) is qMuTau else q_mt.exp_tau().expand(N)
    log_L_true_model = 0
    log_L_var_model = 0
    for n in range(N[0]):
        y_n = obs[:, n]
        u_true = true_Z[n]
        u_var = torch.argmax(q_z.pi[n])
        obs_model_true = dist.Normal(true_C[u_true] * true_mu[n], true_tau[n])
        obs_model_var = dist.Normal(max_prob_cat[u_var] * exp_var_mu[n], exp_var_tau[n])

        log_L_true_model += obs_model_true.log_prob(y_n).sum()
        log_L_var_model += obs_model_var.log_prob(y_n).sum()

    print(f"tot log_L_true_model: {log_L_true_model.sum():,}")
    print(f"tot log_L_var_model: {log_L_var_model.sum():,}")


def fixed_T_comparisons(obs, true_C, true_Z, true_pi, true_mu, true_tau, true_epsilon, q_c: qC, q_z: qZ, qpi: qPi, q_mt: qMuTau):
    compare_qC_and_true_C(true_C, q_c, threshold=50)
    compare_qZ_and_true_Z(true_Z, q_z)
    compare_qMuTau_with_true_mu_and_tau(true_mu, true_tau, q_mt)
    compare_obs_likelihood_under_true_vs_var_model(obs, true_C, true_Z, true_mu, true_tau, q_c, q_z, q_mt)

