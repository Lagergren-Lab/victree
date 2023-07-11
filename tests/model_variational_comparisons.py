import itertools

import torch
import torch.distributions as dist
import torch.nn.functional as f
from sklearn.metrics import adjusted_rand_score

from variational_distributions.observational_variational_distribution import qPsi
from variational_distributions.var_dists import qC, qZ, qPi, qMuTau, qEpsilonMulti


def detect_qC_shifts(true_C, q_c_max):
    K, M = true_C.shape
    diff = (true_C - q_c_max).float()
    mean_diff = torch.mean(diff, dim=-1)
    diff_minus_mean = torch.abs(diff - mean_diff.expand(M, K).T)
    n_shifted = 0
    for k in range(K):
        if torch.sum(diff_minus_mean[k]) < 0.1 and torch.sum(torch.abs(diff[k])) > (M * 0.9):
            print(f"Shift detected of clone {k} (adjusted labeling), qC: {q_c_max[k]}")
            print(f"True c: {true_C[k]}")
            n_shifted += M
    return n_shifted

def compare_qC_and_true_C(true_C, q_c: qC, qz_perm=None, threshold=10):
    """
    Compares the argmax of the qC categorical distribution with the true C used to generate the data.
    :param true_C:
    :param q_c:
    :param threshold:
    :return:
    """
    K, M = true_C.shape
    marginals = q_c.single_filtering_probs
    perms = []
    if qz_perm is None:
        perms_non_root = list(itertools.permutations(range(1, K)))
        for perm in perms_non_root:
            perms.append([0] + list(perm))
    else:
        perms.append(qz_perm)

    max_prob_cat = torch.argmax(marginals, dim=-1)  # Account for label switching
    if K < 8:  # memory issues for larger K
        n_diff = torch.min((max_prob_cat[perms, :] != true_C).sum(2).sum(1))
        best_perm_idx = torch.argmin((max_prob_cat[perms, :] != true_C).sum(2).sum(1))
    else:
        print(f"Start q(C) evaluation w.r.t. label switching on {len(perms)} permutations")
        n_diff = K * M
        for i, perm in enumerate(perms):
            n_diff_i = (max_prob_cat[perm, :] != true_C).sum()
            if n_diff_i < n_diff:
                best_perm_idx = i
                n_diff = n_diff_i

    n_shifted = detect_qC_shifts(true_C, max_prob_cat[perms[best_perm_idx], :]) if n_diff / (K*M) > 0.2 else 0
    print(f"Number of different true C and argmax(q(C)): {n_diff} ({n_shifted} shifted) out of {K*M} states")


def compare_qZ_and_true_Z(true_Z, q_z: qZ):
    """
    Compares the argmax of the parameters of qZ, i.e., the categorical probabilites, with the true cell-to-clone assignments used to generate the data.
    :param true_Z: True cell-to-clone assignments.
    :param q_z: qZ variational distribtion object.
    :return:
    """
    print(f"------- Compare qZ clustering -----------")
    N, K = q_z.pi.shape
    perms = []
    perms_non_root = list(itertools.permutations(range(1, K)))
    max_prob_qZ = torch.argmax(q_z.pi, dim=1)
    for perm in perms_non_root:
        perms.append([0] + list(perm))
    n_perms = len(perms)
    print(f"Start q(Z) evaluation w.r.t. label switching on {n_perms} permutations")
    if n_perms > 10**7:
        raise RuntimeWarning("Large number of permutation might take very long execution time.")
        # TODO: Implement DP-like algorithm for finding best permutation
    accuracy_best = 0
    for perm in perms:
        max_prob_qZ_perm = torch.tensor([perm[i] for i in max_prob_qZ])
        accuracy = torch.eq(max_prob_qZ_perm, true_Z).sum() / N
        if accuracy > accuracy_best:
            best_perm = perm
            accuracy_best = accuracy

    ari = adjusted_rand_score(max_prob_qZ, true_Z)
    print(f"Adjust rand score between q(Z) and true Z: {ari}")
    print(f"Accuracy q(Z) and true Z: {accuracy_best}")
    print(f"Adjusted labeling: {best_perm}")
    print(f"----------------------------------------------")
    return ari, best_perm, accuracy_best


def compare_qMuTau_with_true_mu_and_tau(true_mu, true_tau, q_mt):
    N = true_mu.shape[0]
    square_dist_mu = torch.pow(true_mu - q_mt.nu, 2)
    print(f"MEAN square dist mu: {square_dist_mu.mean()} +- ({square_dist_mu.std()})")
    print(f"MAX square dist mu: {square_dist_mu.max()}")

    square_dist_tau = torch.pow(true_tau - q_mt.exp_tau(), 2)
    print(f"MEAN square dist tau: {square_dist_tau.mean()} +- ({square_dist_tau.std()})")
    print(f"MAX square dist tau: {square_dist_tau.max()}")
    if N > 5:
        largest_error_cells_idx = torch.topk(square_dist_mu.flatten(), 5).indices
        return largest_error_cells_idx


def compare_particular_cells(cells_idx, true_mu, true_tau, true_C, true_Z, q_mt: qMuTau, q_c: qC, q_z: qZ):
    print(f"------- Compare particular cells {cells_idx} -----------")
    torch.set_printoptions(precision=2)
    q_exp_tau = q_mt.exp_tau()
    K, M = true_C.shape
    m_1 = int(M/2)
    m_2 = int(M/2) + 10
    for cell in cells_idx:
        true_clone_idx = true_Z[cell]
        q_clone_idx = q_z.pi[cell].argmax()
        q_C_max = q_c.single_filtering_probs[q_clone_idx].argmax(-1)[m_1:m_2]
        print(f"Cell {cell}: true mu {true_mu[cell]} tau {true_tau[cell]} C {true_C[true_clone_idx, m_1:m_2]} (clone {true_clone_idx})")
        print(f"Cell {cell}: q(mu) {q_mt.nu[cell]} q(tau) {q_exp_tau[cell]} q(C) {q_C_max} (clone {q_clone_idx})")


def compare_obs_likelihood_under_true_vs_var_model(obs, true_C, true_Z, true_mu, true_tau, q_c, q_z, q_mt, best_perm):
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


def compare_qEpsilon_and_true_epsilon(true_epsilon, q_epsilon: qEpsilonMulti):
    q_means = q_epsilon.mean()
    q_variance = q_epsilon.var()
    print(f"-------------- qEpsilonMulti evaluations ---------")
    for key in true_epsilon.keys():
        print(f"Diff true epsilon E_q[epsilon] for edge {key}: {torch.abs(true_epsilon[key] - q_means[key])}")
        print(f"Var_q[epsilon] for edge {key} {q_variance[key]}")


def fixed_T_comparisons(obs, true_C, true_Z, true_pi, true_mu, true_tau, true_epsilon,
                        q_c: qC, q_z: qZ, qpi: qPi, q_mt: qMuTau, q_eps: qEpsilonMulti = None):
    torch.set_printoptions(precision=2)
    ari, perm, acc = compare_qZ_and_true_Z(true_Z, q_z)
    compare_qC_and_true_C(true_C, q_c, qz_perm=perm, threshold=50)
    cell_idxs = compare_qMuTau_with_true_mu_and_tau(true_mu, true_tau, q_mt)
    compare_particular_cells(cell_idxs, true_mu, true_tau, true_C, true_Z, q_mt, q_c, q_z)
    if q_eps is not None:
        compare_qEpsilon_and_true_epsilon(true_epsilon, q_eps)
    compare_obs_likelihood_under_true_vs_var_model(obs, true_C, true_Z, true_mu, true_tau, q_c, q_z, q_mt, perm)


def compare_phi_and_true_phi(phi, q_psi, perm):
    K = phi.shape[0]
    for k in range(K):
        print(f"Clone {k} - true: {phi[k]} - var: {q_psi.phi[perm[k]]}")

def fixed_T_urn_model_comparisons(x, R, gc, phi, true_C, true_Z, true_pi, true_epsilon,
                        q_c: qC, q_z: qZ, qpi: qPi, q_psi: qPsi, q_eps: qEpsilonMulti = None):
    torch.set_printoptions(precision=2)
    ari, perm, acc = compare_qZ_and_true_Z(true_Z, q_z)
    compare_qC_and_true_C(true_C, q_c, qz_perm=perm, threshold=50)
    compare_phi_and_true_phi(phi, q_psi, perm)
    if q_eps is not None:
        compare_qEpsilon_and_true_epsilon(true_epsilon, q_eps)
