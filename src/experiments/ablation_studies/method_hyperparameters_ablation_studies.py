import copy
import os
import pathlib

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

import simul
import utils
from inference.victree import VICTree
from utils import tree_utils, visualization_utils
from utils.config import Config
from variational_distributions.joint_dists import VarTreeJointDist
from variational_distributions.var_dists import qC, qT, qEpsilonMulti, qZ, qPi, qMuTau


def set_up_q(config, eps_a_prior, eps_beta_prior, pi_delta_prior, mt_nu_prior, mt_lmbda_prior, mt_alpha_prior,
             mt_beta_prior):
    qc = qC(config)
    qt = qT(config)
    qeps = qEpsilonMulti(config, alpha_prior=eps_a_prior, beta_prior=eps_beta_prior)
    qz = qZ(config)
    qpi = qPi(config, delta_prior=pi_delta_prior)
    qmt = qMuTau(config, nu_prior=mt_nu_prior, lambda_prior=mt_lmbda_prior, alpha_prior=mt_alpha_prior,
                 beta_prior=mt_beta_prior)
    return qc, qt, qeps, qz, qpi, qmt


def init_q_to_prior_parameters(q, mt_nu_prior, mt_lmbda_prior,
                               mt_alpha_prior, mt_beta_prior):
    q.initialize()
    q.eps.initialize('prior')
    q.pi.initialize('prior')
    q.mt.initialize('fixed', loc=mt_nu_prior, precision_factor=mt_lmbda_prior, shape=mt_alpha_prior, rate=mt_beta_prior)


def create_output_catalog(ablation_study_specific_str):
    dirs = os.getcwd().split('/')
    dir_top_idx = dirs.index('ablation_studies')
    dir_path = dirs[dir_top_idx:]
    path = os.path.join(*dir_path)
    path = os.path.join(path, ablation_study_specific_str)
    base_dir = '../../../output'
    full_path = os.path.join(base_dir, path)
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    return full_path


def step_size_ablation_study(save=False):
    utils.config.set_seed(0)
    inference_seeds = list(range(0, 5))
    step_sizes = [1., 0.5, 0.3, 0.1, 0.05, 0.01, 0.005]
    n_iter = 500
    dir_delta0 = 10.
    nu_0 = 1.
    lambda_0 = 10.
    alpha0 = 500.
    beta0 = 50.
    a0 = 1.0
    b0 = 500.0
    N, M, K, A = (1000, 3000, 8, 7)

    tree = tree_utils.generate_fixed_tree(K)

    # Generate data
    config = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M)
    out = simul.simulate_full_dataset(config=config, eps_a=a0, eps_b=b0, mu0=nu_0, lambda0=lambda_0, alpha0=alpha0,
                                      beta0=beta0, dir_delta=dir_delta0, tree=tree)
    y, c, z, pi, mu, tau, eps, eps0, chr_idx = (out['obs'], out['c'], out['z'], out['pi'], out['mu'], out['tau'],
                                                out['eps'], out['eps0'], out['chr_idx'])
    print(f"------------ Data set sanity check ------------")
    if save:
        experiment_str = f'step_size/N{N}_M{M}_{K}_{A}_nIter{n_iter}'
        save_path = create_output_catalog(ablation_study_specific_str=experiment_str)
        visualization_utils.visualize_copy_number_profiles(c, save_path=save_path)
        np.save(save_path + '/eps_simulated', np.array(eps))
        np.save(save_path + '/pi_simulated', np.array(pi))
        np.save(save_path + '/mu_simulated', np.array(mu))
        np.save(save_path + '/tau_simulated', np.array(tau))

    print(f"pi: {pi}")
    print(f"eps: {eps}")
    print(f"Mu in range: [{mu.min()}, {mu.max()}] ")
    print(f"Tau in range: [{tau.min()}, {tau.max()}] ")

    ari_list = []
    ari_std_list = []
    elbo_list = []
    elbo_std_list = []
    for step_size in step_sizes:

        config = copy.deepcopy(config)
        config.step_size = step_size

        ari = []
        elbo = []
        for seed in inference_seeds:
            utils.config.set_seed(seed)

            # Re-initialize q
            qc, qt, qeps, qz, qpi, qmt = set_up_q(config, eps_a_prior=a0, eps_beta_prior=b0, pi_delta_prior=dir_delta0,
                                                  mt_nu_prior=nu_0, mt_lmbda_prior=lambda_0, mt_alpha_prior=alpha0,
                                                  mt_beta_prior=beta0)
            q = VarTreeJointDist(config, y, qc, qz, qt, qeps, qmt, qpi)
            init_q_to_prior_parameters(q, mt_nu_prior=nu_0, mt_lmbda_prior=lambda_0, mt_alpha_prior=alpha0,
                                       mt_beta_prior=beta0)
            copy_tree = VICTree(config, q, y)

            copy_tree.run(n_iter=n_iter)

            ari_seed = adjusted_rand_score(z, copy_tree.q.z.pi.argmax(dim=-1))
            ari.append(ari_seed)
            elbo_seed = copy_tree.elbo
            elbo.append(elbo_seed)
            print(f"ARI for step size {step_size} and seed {seed}: {ari_seed}")
            print(f"ELBO for step size {step_size} and seed {seed}: {elbo_seed}")

        ari_list.append(np.array(ari).mean())
        ari_std_list.append(np.array(ari).std())
        elbo_list.append(np.array(elbo).mean())
        elbo_std_list.append(np.array(elbo).std())
        print(f"mean ARI for step size {step_size}: {np.array(ari).mean()} ({np.array(ari).std()})")
        print(f"mean ELBO for steps size {step_size}: {np.array(elbo).mean()} ({np.array(elbo).std()})")

        if save:
            np.save(save_path + '/' + 'elbo_mean', ari_list)
            np.save(save_path + '/' + 'elbo_std', ari_std_list)
            np.save(save_path + '/' + 'ari_mean', elbo_list)
            np.save(save_path + '/' + 'ari_std', elbo_std_list)
            np.save(save_path + '/' + 'step_sizes', step_sizes)


if __name__ == '__main__':
    step_size_ablation_study(save=True)
