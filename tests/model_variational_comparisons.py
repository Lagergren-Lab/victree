import torch
import torch.distributions as dist
import torch.nn.functional as f

from variational_distributions.var_dists import qC, qZ, qPi, qMuTau


def compare_qC_and_true_C(true_C, q_c: qC, threshold=10):
    """
    Compares the argmax of the qC categorical distribution with the true C used to generate the data.
    :param true_C:
    :param q_c:
    :param threshold:
    :return:
    """
    marginals = q_c.single_filtering_probs
    max_prob_cat = torch.argmax(marginals, dim=-1)
    n_diff = torch.sum(true_C != max_prob_cat)
    print(f"Number of different true C and argmax(q(C)): {n_diff}")
    #assert n_diff <= threshold, f"Number of different true C and argmax(q(C)): {n_diff}"


def compare_qZ_and_true_Z(true_Z, q_z: qZ):
    """
    Compares the argmax of the parameters of qZ, i.e., the categorical probabilites, with the true cell-to-clone assignments used to generate the data.
    :param true_Z: True cell-to-clone assignments.
    :param q_z: qZ variational distribtion object.
    :return:
    """
    N, K = q_z.pi.shape
    z_true_one_hot = f.one_hot(true_Z.long(), num_classes=K)

    total_misclassifications = torch.sum(true_Z.float() != torch.argmax(q_z.pi, dim=1))
    print(f"Total number of mis-classifications: {total_misclassifications}")
    if total_misclassifications != 0:
        n_prints = 5
        mis_class_idx = torch.where(true_Z.float() != torch.argmax(q_z.pi, dim=1))
        rand_idx = torch.randperm(total_misclassifications)
        mis_class_shuffled = mis_class_idx[0][rand_idx]
        i = 0
        while (i < n_prints and i < total_misclassifications):
            print(f"Misclassification of cell {mis_class_shuffled[i]}: qZ = {q_z.pi[mis_class_shuffled[i]]}, pZ = {true_Z[mis_class_shuffled[i]]}")
            i += 1


def compare_qMuTau_with_true_mu_and_tau(true_mu, true_tau, q_mt):
    N = true_mu.shape
    square_dist_mu = torch.pow(true_mu - q_mt.nu, 2)
    print(f"mean square dist mu: {square_dist_mu.mean()} +- ({square_dist_mu.std()})")

    square_dist_tau = torch.pow(true_tau - q_mt.exp_tau(), 2)
    print(f"mean square dist tau: {square_dist_tau.mean()} +- ({square_dist_tau.std()})")


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
    exp_var_tau = q_mt.exp_tau()
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

