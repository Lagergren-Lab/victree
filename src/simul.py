#!/usr/bin/env python3

"""
Data simulation script.
Uses Pyro-PPL for modelling the data only with data generation purposes.
"""
# FIXME: data simulation is not working, need to fix pair_cpd table,
#   maybe C_dict needs to store plain python variables and not pyro variables

import argparse
import logging
import os.path
import random

import h5py
import networkx as nx
from networkx.algorithms.tree.coding import NotATree
from networkx.algorithms.tree.recognition import is_arborescence
import numpy as np
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from matplotlib import pyplot as plt

from variational_distributions.joint_dists import VarTreeJointDist
from utils import tree_utils
from utils.config import Config, set_seed
from utils.eps_utils import TreeHMM, h_eps, h_eps0
from utils.tree_utils import generate_fixed_tree
from variational_distributions.var_dists import qC, qZ, qEpsilonMulti, qMuTau, qPi, qT


def model_tree_markov(data, n_cells, n_sites, n_copy_states, tree: nx.DiGraph,
                      mu_0=torch.tensor(10.),
                      nu=torch.tensor(.1),
                      a0=torch.tensor(1.0),
                      b0=torch.tensor(1.0),
                      alpha0=torch.tensor(10.),
                      beta0=torch.tensor(40.),
                      ):
    # check that tree is with maximum in-degree =1
    if not is_arborescence(tree):
        raise NotATree("The provided graph is not a tree/arborescence")

    n_nodes = len(tree.nodes)

    # PRIORS PARAMETERS

    sigma2 = pyro.sample("sigma", dist.InverseGamma(a0, b0))
    mu = pyro.sample("mu", dist.Normal(mu_0, sigma2 * nu).expand([n_cells]))
    eps = pyro.sample("eps", dist.Beta(alpha0, beta0))

    treeHMM = TreeHMM(n_copy_states, eps=eps)
    cpd = treeHMM.cpd_table
    pair_cpd = treeHMM.cpd_pair_table

    # cell assignments
    # dirichlet param vector (uniform)
    dir_alpha = torch.ones(n_nodes)

    pi = pyro.sample("pi", dist.Dirichlet(dir_alpha))
    z = pyro.sample("z", dist.Categorical(pi).expand([n_cells]))

    # variables to store complete data in
    C_dict = {}
    y = torch.zeros(n_sites, n_cells)

    # dist for root note (dirac on C_r_m)
    prior_tensor = torch.eye(n_copy_states)
    # C_r_m determines the value of the root copy number
    # which is the same for each site
    C_r_m = 2

    for u in nx.dfs_preorder_nodes(tree):
        # root node
        if u == 0:
            for m in range(n_sites):
                # root node is always 2
                C_r_m = pyro.sample("C_{}_{}".format(0, m), dist.Categorical(prior_tensor[C_r_m, :]))

                C_dict[0, m] = C_r_m
                y[m, z == u] = pyro.sample("y_{}_{}".format(0, m), dist.Normal(mu[z == u] * C_r_m, sigma2.sqrt()),
                                           obs=data[m, z == u])
        # inner nodes
        else:
            p = [pred for pred in tree.predecessors(u)][0]
            # no need for pyro.markov, range is equivalent
            # a simple for loop is used in the tutorials for sequential dependencies
            # ref: https://pyro.ai/examples/svi_part_ii.html#Sequential-plate
            for m in range(n_sites):
                # current site, parent copy number is always available
                C_p_m = int(C_dict[p, m])

                zipping_dist = None
                if m == 0:
                    # initial state case only depends on the parent initial state
                    # use pair_cpd as m-1 is not available
                    zipping_dist = dist.Categorical(torch.tensor([pair_cpd[i, C_p_m] for i in range(n_copy_states)]))
                else:
                    # previous copy numbers
                    C_p_m_1 = int(C_dict[p, m - 1])
                    C_u_m_1 = int(C_dict[u, m - 1])

                    # other states depend on 3 states
                    zipping_dist = dist.Categorical(
                        probs=torch.tensor([cpd[i, C_u_m_1, C_p_m, C_p_m_1] for i in range(n_copy_states)]))

                C_u_m = pyro.sample("C_{}_{}".format(u, m), zipping_dist)

                # save values in dict
                C_dict[u, m] = C_u_m
                y[m, z == u] = pyro.sample("y_{}_{}".format(u, m), dist.Normal(mu[z == u] * C_u_m, sigma2.sqrt()),
                                           obs=data[m, z == u])

    C = torch.empty((n_nodes, n_sites))
    for u, m in C_dict.keys():
        C[u, m] = C_dict[u, m]

    return C, y, z, pi, mu, sigma2, eps


def model_tree_markov_full(data, n_cells, n_sites, n_copy_states, tree: nx.DiGraph,
                           mu_0=torch.tensor(10.),
                           lambda_0=torch.tensor(.1),
                           alpha0=torch.tensor(1.),
                           beta0=torch.tensor(1.),
                           a0=torch.tensor(1.0),
                           b0=torch.tensor(20.0),
                           dir_alpha0=torch.tensor(1.0),
                           ):
    # check that tree is with maximum in-degree =1
    if not is_arborescence(tree):
        raise NotATree("The provided graph is not a tree/arborescence")

    n_nodes = len(tree.nodes)

    # PRIORS PARAMETERS

    # cell assignments
    # dirichlet param vector (uniform)
    dir_alpha = torch.ones(n_nodes) * dir_alpha0
    pi = pyro.sample("pi", dist.Dirichlet(dir_alpha))

    # Per cell variables
    with pyro.plate("cells", n_cells) as n:
        tau = pyro.sample("tau_{}".format(n), dist.Gamma(alpha0, beta0))
        mu = pyro.sample("mu_{}".format(n), dist.Normal(mu_0, 1. / (lambda_0 * tau).sqrt()))
        z = pyro.sample("z_{}".format(n), dist.Categorical(pi))

    eps = pyro.sample("eps", dist.Beta(a0, b0))

    treeHMM = TreeHMM(n_copy_states, eps=eps)
    cpd = treeHMM.cpd_table
    pair_cpd = treeHMM.cpd_pair_table

    # variables to store complete data in
    C_dict = {}
    y = torch.zeros(n_sites, n_cells)

    # dist for root note (dirac on C_r_m)
    prior_tensor = torch.eye(n_copy_states)
    # C_r_m determines the value of the root copy number
    # which is the same for each site
    C_r_m = 2

    for u in nx.dfs_preorder_nodes(tree):
        # root node
        if u == 0:
            for m in range(n_sites):
                # root node is always 2
                C_r_m = pyro.sample("C_{}_{}".format(0, m), dist.Categorical(prior_tensor[C_r_m, :]))

                C_dict[0, m] = C_r_m
                ind = torch.where(z == u)[0]
                for n in ind:
                    y[m, n] = pyro.sample("y_{}_{}".format(m, n),
                                          dist.Normal(mu[n] * C_r_m, 1.0 / (lambda_0 * tau[n].sqrt())), obs=data[m, n])
        # inner nodes
        else:
            p = [pred for pred in tree.predecessors(u)][0]
            # no need for pyro.markov, range is equivalent
            # a simple for loop is used in the tutorials for sequential dependencies
            # ref: https://pyro.ai/examples/svi_part_ii.html#Sequential-plate
            for m in range(n_sites):
                # current site, parent copy number is always available
                C_p_m = int(C_dict[p, m])

                zipping_dist = None
                if m == 0:
                    # initial state case only depends on the parent initial state
                    # use pair_cpd as m-1 is not available
                    zipping_dist = dist.Categorical(torch.tensor([pair_cpd[i, C_p_m] for i in range(n_copy_states)]))
                else:
                    # previous copy numbers
                    C_p_m_1 = int(C_dict[p, m - 1])
                    C_u_m_1 = int(C_dict[u, m - 1])

                    # other states depend on 3 states
                    zipping_dist = dist.Categorical(
                        probs=torch.tensor([cpd[i, C_u_m_1, C_p_m, C_p_m_1] for i in range(n_copy_states)]))

                C_u_m = pyro.sample("C_{}_{}".format(u, m), zipping_dist)

                # save values in dict
                C_dict[u, m] = C_u_m
                ind = torch.where(z == u)[0]
                y[m, ind] = pyro.sample("y_{}_{}".format(u, ind), dist.Normal(mu[ind] * C_u_m, 1.0 / tau[ind].sqrt()),
                                        obs=data[m, ind])

    C = torch.empty((n_nodes, n_sites))
    for u, m in C_dict.keys():
        C[u, m] = C_dict[u, m]

    return C, y, z, pi, mu, tau, eps


def sample_raw_counts_from_corrected_data(obs):
    # TODO: generate counts from observation using DirichletMultinomial
    #   trying to avoid computationally expensive sampling i.e. sampling high dimensional
    #   multinomial might be costly but if each bin is sampled individually with BetaBinomial
    #   it might be better
    # temporary solution, sample from poisson with mean rho * obs
    rho = 300.
    raw_counts = torch.distributions.Poisson(torch.clamp(obs * rho, min=0.)).sample((1,))[0]
    assert raw_counts.shape == obs.shape
    return raw_counts


def simulate_full_dataset(config: Config, eps_a=5., eps_b=50., mu0=1., lambda0=10.,
                          alpha0=500., beta0=50., dir_alpha: [float | list[float]] = 1., tree=None, raw_reads=True):
    """
Generate full simulated dataset.
    Args:
        config: configuration object
        eps_a: float, param for Beta distribution over epsilon
        eps_b: float, param for Beta distribution over epsilon
        mu0: float, param for NormalGamma distribution over mu/tau
        lambda0: float, param for NormalGamma distribution over mu/tau
        alpha0: float, param for NormalGamma distribution over mu/tau
        beta0: float, param for NormalGamma distribution over mu/tau

    Returns:
        dictionary with keys: ['obs', 'c', 'z', 'pi', 'mu', 'tau', 'eps', 'eps0', 'tree']
    """
    # generate random tree
    tree = nx.random_tree(config.n_nodes, create_using=nx.DiGraph) if tree is None else tree
    logging.debug(f'sampled tree: {tree_utils.tree_to_newick(tree)}')
    # generate eps from Beta(a, b)
    eps = {}
    for u, v in tree.edges:
        eps[u, v] = torch.distributions.Beta(eps_a, eps_b).sample()
        tree.edges[u, v]['weight'] = eps[u, v]
    eps0 = config.eps0
    # generate copy numbers
    c = torch.empty((config.n_nodes, config.chain_length), dtype=torch.long)
    c[0, :] = 2 * torch.ones(config.chain_length)
    h_eps0_cached = h_eps0(config.n_states, eps0)
    for u, v in nx.bfs_edges(tree, source=0):
        t0 = h_eps0_cached[c[u, 0], :]
        c[v, 0] = torch.distributions.Categorical(probs=t0).sample()
        h_eps_uv = h_eps(config.n_states, eps[u, v])
        for m in range(1, config.chain_length):
            # j', j, i', i
            transition = h_eps_uv[:, c[v, m - 1], c[u, m], c[u, m - 1]]
            c[v, m] = torch.distributions.Categorical(probs=transition).sample()

    # sample mu_n, tau_n
    tau = torch.distributions.Gamma(alpha0, beta0).sample((config.n_cells,))
    mu = torch.distributions.Normal(mu0, 1. / torch.sqrt(lambda0 * tau)).sample()
    assert mu.shape == tau.shape
    # sample assignments
    if isinstance(dir_alpha, float):
        dir_alpha_tensor = torch.ones(config.n_nodes) * dir_alpha
    elif isinstance(dir_alpha, list):
        dir_alpha_tensor = torch.tensor(dir_alpha)
    else:
        raise ValueError(f"dir_alpha param must be either a k-size list of float or a float (not {type(dir_alpha)})")
    pi = torch.distributions.Dirichlet(dir_alpha_tensor).sample()
    z = torch.distributions.Categorical(pi).sample((config.n_cells,))
    # sample observations
    obs_mean = c[z, :] * mu[:, None]  # n_cells x chain_length
    scale_expanded = torch.pow(tau, -1 / 2).reshape(-1, 1).expand(-1, config.chain_length)
    # (chain_length x n_cells)
    obs = torch.distributions.Normal(obs_mean, scale_expanded).sample()
    obs = obs.T
    assert obs.shape == (config.chain_length, config.n_cells)

    raw_counts = sample_raw_counts_from_corrected_data(obs) if raw_reads is True else None
    out_simul = {
        'obs': obs,
        'raw': raw_counts,
        'c': c,
        'z': z,
        'pi': pi,
        'mu': mu,
        'tau': tau,
        'eps': eps,
        'eps0': eps0,
        'tree': tree
    }
    return out_simul


def simulate_copy_tree_data(K, M, A, tree: nx.DiGraph, eps_a, eps_b, eps_0):
    eps = {}
    logging.debug(f'Copy Tree data simulation - eps_a: {eps_a}, eps_b: {eps_b}, eps_0:{eps_0} ')
    eps_dist = torch.distributions.Beta(eps_a, eps_b)
    if eps_dist.variance > 0.1 * eps_dist.mean:
        logging.warning(f'Large variance for epsilon: {eps_dist.variance} (mean: {eps_dist.mean}. Consider increasing '
                        f'eps_b param.')

    for u, v in tree.edges:
        eps[u, v] = eps_dist.sample()
        tree.edges[u, v]['weight'] = eps[u, v]

    # generate copy numbers
    c = torch.empty((K, M), dtype=torch.long)
    c[0, :] = 2 * torch.ones(M)
    h_eps0_cached = h_eps0(A, eps_0)
    for u, v in nx.bfs_edges(tree, source=0):
        t0 = h_eps0_cached[c[u, 0], :]
        c[v, 0] = torch.distributions.Categorical(probs=t0).sample()
        h_eps_uv = h_eps(A, eps[u, v])
        for m in range(1, M):
            # j', j, i', i
            transition = h_eps_uv[:, c[v, m - 1], c[u, m], c[u, m - 1]]
            c[v, m] = torch.distributions.Categorical(probs=transition).sample()

    return eps, c


def simulate_component_assignments(N, K, dir_delta):
    logging.debug(f'Component assignments simulation - delta: {dir_delta}')
    if isinstance(dir_delta, float):
        dir_alpha_tensor = torch.ones(K) * dir_delta
    else:
        dir_alpha_tensor = torch.tensor(dir_delta)
    pi = torch.distributions.Dirichlet(dir_alpha_tensor).sample()
    z = torch.distributions.Categorical(pi).sample((N,))
    return pi, z


def simulate_Psi_LogNormal(N, alpha0, beta0):
    tau_dist = dist.Gamma(alpha0, beta0)
    tau = tau_dist.sample((N,))
    return tau


def simulate_observations_LogNormal(N, M, c, z, R, gc, tau):
    x = torch.zeros((N, M))
    phi = torch.einsum("km, m -> k", c.float(), gc)
    for n in range(N):
        k = z[n]
        mu_n = c[k] * gc * R[n] / phi[k]
        x_n_dist = dist.LogNormal(mu_n, 1. / tau[n])
        x[n] = x_n_dist.sample()

    return x, phi


def simulate_observations_Poisson(N, M, c, z, R, gc):
    x = torch.zeros((N, M))
    phi = torch.einsum("km, m -> k", c.float(), gc)
    for n in range(N):
        k = z[n]
        lmbda_n = c[k] * gc * R[n] / phi[k]
        x_n_dist = dist.Poisson(lmbda_n)
        x[n] = x_n_dist.sample()

    return x, phi


def simulate_total_reads(N, R_0):
    a = int(R_0 - R_0 / 10.)
    b = int(R_0 + R_0 / 10.)
    logging.debug(f"Reads per cell simulation: R in  [{a},{b}] ")
    R = torch.randint(a, b, (N,))
    return R


def simulate_gc_site_corrections(M):
    a = 0.8
    b = 1.0
    logging.debug(f"GC correction per site simulation: g_m in  [{a},{b}] ")
    gc_dist = dist.Uniform(a, b)
    gc = gc_dist.sample((M,))
    return gc


def simulate_data_total_GC_urn_model(tree, N, M, K, A, R_0, emission_model="poisson", eps_a=5., eps_b=50., eps_0=1.,
                                     alpha0=500., beta0=50., dir_delta: [float | list[float]] = 1.):
    """
    Generate full simulated dataset.
    Args:
        config: configuration object
        eps_a: float, param for Beta distribution over epsilon
        eps_b: float, param for Beta distribution over epsilon
        mu0: float, param for NormalGamma distribution over mu/tau
        alpha0: float, param for NormalGamma distribution over mu/tau
        beta0: float, param for NormalGamma distribution over mu/tau

    Returns:
        dictionary with keys: ['obs', 'c', 'z', 'pi', 'mu', 'tau', 'eps', 'eps0', 'tree']

    """
    # generate random tree
    tree = nx.random_tree(K, create_using=nx.DiGraph) if tree is None else tree
    logging.debug(f'sampled tree: {tree_utils.tree_to_newick(tree)}')

    # Copy tree simulation
    eps, c = simulate_copy_tree_data(K, M, A, tree, eps_a, eps_b, eps_0)

    # Mixture model associated simulations
    pi, z = simulate_component_assignments(N, K, dir_delta)

    # Component variables and observations
    R = simulate_total_reads(N, R_0)
    gc = simulate_gc_site_corrections(M)

    if emission_model.lower() == "lognormal":
        psi = simulate_Psi_LogNormal(N, alpha0, beta0)
        x, phi = simulate_observations_LogNormal(N, M, c, z, R, gc, psi)
    elif emission_model.lower() == "poisson":
        psi = None
        x, phi = simulate_observations_Poisson(N, M, c, z, R, gc)


    out_simul = {
        'x': x,
        'R': R,
        'gc': gc,
        'c': c,
        'z': z,
        'phi': phi,
        'pi': pi,
        'psi': psi,
        'eps': eps,
        'eps0': eps_0,
        'tree': tree
    }
    return out_simul


def model_tree_markov_fixed_parameters(data, n_cells, n_sites, n_copy_states, tree: nx.DiGraph,
                                       mu: torch.Tensor,
                                       tau: torch.Tensor,
                                       pi: torch.Tensor,
                                       eps: torch.Tensor
                                       ):
    # check that tree is with maximum in-degree =1
    if not is_arborescence(tree):
        raise NotATree("The provided graph is not a tree/arborescence")

    n_nodes = len(tree.nodes)

    # PRIORS PARAMETERS

    # cell assignments
    # dirichlet param vector (uniform)
    pi_param = pyro.param("pi", pi)

    # Per cell variables
    with pyro.plate("cells", n_cells) as n:
        tau_param = pyro.param("tau_{}".format(n), mu[n])
        mu_param = pyro.param("mu_{}".format(n), tau[n])
        z = pyro.sample("z_{}".format(n), dist.Categorical(pi))

    eps_param = pyro.param("eps", eps)

    treeHMM = TreeHMM(n_copy_states, eps=eps)
    cpd = treeHMM.cpd_table
    pair_cpd = treeHMM.cpd_pair_table

    # variables to store complete data in
    C_dict = {}
    y = torch.zeros(n_sites, n_cells)

    # dist for root note (dirac on C_r_m)
    prior_tensor = torch.eye(n_copy_states)
    # C_r_m determines the value of the root copy number
    # which is the same for each site
    C_r_m = 2

    for u in nx.dfs_preorder_nodes(tree):
        # root node
        if u == 0:
            for m in range(n_sites):
                # root node is always 2
                C_r_m = pyro.sample("C_{}_{}".format(0, m), dist.Categorical(prior_tensor[C_r_m, :]))

                C_dict[0, m] = C_r_m
                ind = torch.where(z == u)[0]
                for n in ind:
                    y[m, n] = pyro.sample("y_{}_{}".format(m, n), dist.Normal(mu[n] * C_r_m, 1.0 / tau[n].sqrt()),
                                          obs=data[m, n])
        # inner nodes
        else:
            p = [pred for pred in tree.predecessors(u)][0]
            # no need for pyro.markov, range is equivalent
            # a simple for loop is used in the tutorials for sequential dependencies
            # ref: https://pyro.ai/examples/svi_part_ii.html#Sequential-plate
            for m in range(n_sites):
                # current site, parent copy number is always available
                C_p_m = int(C_dict[p, m])

                zipping_dist = None
                if m == 0:
                    # initial state case only depends on the parent initial state
                    # use pair_cpd as m-1 is not available
                    zipping_dist = dist.Categorical(torch.tensor([pair_cpd[i, C_p_m] for i in range(n_copy_states)]))
                else:
                    # previous copy numbers
                    C_p_m_1 = int(C_dict[p, m - 1])
                    C_u_m_1 = int(C_dict[u, m - 1])

                    # other states depend on 3 states
                    zipping_dist = dist.Categorical(
                        probs=torch.tensor([cpd[i, C_u_m_1, C_p_m, C_p_m_1] for i in range(n_copy_states)]))

                C_u_m = pyro.sample("C_{}_{}".format(u, m), zipping_dist)

                # save values in dict
                C_dict[u, m] = C_u_m
                ind = torch.where(z == u)[0]
                y[m, ind] = pyro.sample("y_{}_{}".format(u, ind), dist.Normal(mu[ind] * C_u_m, 1.0 / (tau[ind].sqrt())),
                                        obs=data[m, ind])

    C = torch.empty((n_nodes, n_sites))
    for u, m in C_dict.keys():
        C[u, m] = C_dict[u, m]

    return C, y, z, pi, mu, tau, eps


def write_simulated_dataset_h5(data, out_dir, filename, gt_mode='h5'):
    """
    Output data structure
    - X: raw reads (dataset) shape (n_cells, chain_length)
    - gt: ground truth(group)
    - layers: group
        - copy-number: dataset shape (n_cells, chain_length)

    Parameters
    ----------
    filename name of simulated dataset (without extension)
    out_dir directory
    data dict, output of simulate_full_dataset, with keys:
    [obs, c, z, pi, mu, tau, eps, eps0, tree]

    Returns
    -------
    """
    f = h5py.File(os.path.join(out_dir, filename + '.h5'), 'w')
    x_ds = f.create_dataset('X', data=data['raw'].T)
    layers_grp = f.create_group('layers')
    z = data['z']
    cn_state = data['c'][z, :]
    assert cn_state.shape == x_ds.shape
    layers_grp.create_dataset('state', data=cn_state)
    layers_grp.create_dataset('copy', data=data['obs'].T)

    k = data['c'].shape[0]
    if gt_mode == 'h5':
        gt_group = f.create_group('gt')
        gt_group.create_dataset('copy', data=data['c'].numpy())
        gt_group.create_dataset('cell_assignment', data=data['z'].numpy())
        gt_group.create_dataset('pi', data=data['pi'].numpy())
        eps_npy = np.zeros((k, k))
        for uv, e in data['eps'].items():
            eps_npy[uv] = e
        gt_group.create_dataset('eps', data=eps_npy)
        # gt_group.create_dataset('tree', data=torch.tensor(list(data['tree'].edges.data('weight'))))
        gt_group.create_dataset('mu', data=data['mu'].numpy())
        gt_group.create_dataset('tau', data=data['tau'].numpy())
    elif gt_mode == 'numpy':
        # write gt as numpy arrays which can be easily read into R
        # with reticulate
        gt_path = os.path.join(out_dir, 'gt_' + filename)
        if not os.path.exists(gt_path):
            os.mkdir(gt_path)
        # copy numbers
        np.save(os.path.join(gt_path, 'copy.npy'), data['c'].numpy())
        # cell assignment
        np.save(os.path.join(gt_path, 'cell_assignment.npy'), data['z'].numpy())
        # pi
        np.save(os.path.join(gt_path, 'pi.npy'), data['pi'].numpy())
        # eps
        eps_npy = np.zeros((k, k))
        for uv, e in data['eps'].items():
            eps_npy[uv] = e
        np.save(os.path.join(gt_path, 'eps.npy'), eps_npy)
        # mu-tau
        np.save(os.path.join(gt_path, 'mu.npy'), data['mu'].numpy())
        np.save(os.path.join(gt_path, 'tau.npy'), data['tau'].numpy())
        # tree can be retrieved from the non-zero values of eps matrix

    f.close()


def write_sample_dataset_h5(dest_path):
    n_cells = 300
    n_sites = 100
    n_copy_states = 5
    n_nodes = 4
    tree = generate_fixed_tree(n_nodes)
    mu_0 = 100.0
    nu_0 = 0.1
    alpha0 = 10.
    beta0 = 40.
    a0 = .5
    b0 = .5
    dir_alpha0 = 1.
    data = torch.ones((n_sites, n_cells))
    unconditioned_model = poutine.uncondition(model_tree_markov_full)
    C, y, z, pi, mu, tau, eps = unconditioned_model(data, n_cells, n_sites, n_copy_states, tree, mu_0, nu_0, alpha0,
                                                    beta0, a0, b0, dir_alpha0)
    # write cn for each cell separately
    f = h5py.File(dest_path, 'w')
    x_ds = f.create_dataset('X', data=y.int().clamp(min=0))
    layers_grp = f.create_group('layers')
    cn_state = C[z, :].T
    assert cn_state.shape == x_ds.shape
    layers_grp.create_dataset('state', data=cn_state)
    f.close()


def main(args):
    if args.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # params
    n_cells = 10
    n_sites = 5
    n_copy_states = 5

    data = torch.ones(n_sites, n_cells)

    tree = nx.DiGraph()
    tree.add_edge(0, 1)
    tree.add_edge(0, 2)
    tree.add_edge(1, 3)
    tree.add_edge(1, 4)

    # draw tree topology
    f = plt.figure()
    nx.draw_networkx(tree, ax=f.add_subplot(111))
    f.savefig("../fig/tree_topology.png")

    # draw bayes net
    graph = pyro.render_model(model_tree_markov, model_args=(data, n_cells, n_sites, n_copy_states, tree,))
    graph.render(outfile='../fig/graph.png')

    logging.info("Simulate data")
    # simulate latent variable as well as observations (synthetic data generation)
    # using "uncondition" handler
    unconditioned_model = poutine.uncondition(model_tree_markov)
    # C_r, C_u, y_u = unconditioned_model(data, n_cells, n_sites, n_copy_states, )
    C, y, z = unconditioned_model(data, n_cells, n_sites, n_copy_states, tree, )
    print(f"tree: {tree_utils.tree_to_newick(tree, 0)}")
    print(f"C: {C}")
    print(f"y: {y}")
    print(f"z: {z}")


def tree_to_newick_old(g: nx.DiGraph, root=None):
    """
    NetworkX tree topology to string in Newick format
    edited code from https://stackoverflow.com/a/57393072/11880992
    """
    # make sure the graph is a tree
    assert is_arborescence(g)
    if root is None:
        roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
        assert 1 == len(roots)
        root = roots[0][0]

    subgs = []
    for child in g[root]:
        if len(g[child]) > 0:
            subgs.append(tree_to_newick_old(g, root=child))
        else:
            subgs.append(str(child))
    return "(" + ','.join(subgs) + ")" + str(root)


def generate_dataset_var_tree(config: Config) -> VarTreeJointDist:
    nu_prior = 1.
    lambda_prior = 100.
    alpha_prior = 500.
    beta_prior = 50.
    simul_data = simulate_full_dataset(config, eps_a=5., eps_b=50., mu0=nu_prior, lambda0=lambda_prior,
                                       alpha0=alpha_prior, beta0=beta_prior)

    fix_qc = qC(config, true_params={
        "c": simul_data['c']
    })

    fix_qz = qZ(config, true_params={
        "z": simul_data['z']
    })

    fix_qeps = qEpsilonMulti(config, true_params={
        "eps": simul_data['eps']
    })

    fix_qmt = qMuTau(config, true_params={
        "mu": simul_data['mu'],
        "tau": simul_data['tau']
    }).initialize()

    fix_qpi = qPi(config, true_params={
        "pi": simul_data['pi']
    })

    fix_qt = qT(config, true_params={
        "tree": simul_data['tree']
    })

    joint_q = VarTreeJointDist(config, simul_data['obs'], fix_qc, fix_qz, fix_qt, fix_qeps, fix_qmt, fix_qpi)
    return joint_q


# script for simulating data
if __name__ == '__main__':
    cli = argparse.ArgumentParser(
        description="Data simulation script. Output format is compatible with VIC-Tree interface."
    )
    cli.add_argument('-o', '--out-path',
                     type=str,
                     default='./datasets', help="output directory e.g. ./datasets")
    cli.add_argument('-K', '--n-nodes',
                     type=int,
                     default=5, help="number of nodes")
    cli.add_argument('-A', '--n-states',
                     type=int,
                     default=7, help="number of copy number states from 0 to A")
    cli.add_argument('-N', '--n-cells',
                     type=int,
                     default=300, help="number of cells")
    cli.add_argument('-M', '--chain-length',
                     type=int,
                     default=1000, help="number of sites in copy number and DNA sequence (seq. length)")
    cli.add_argument('-s', '--seed',
                     type=int,
                     default=42, help="RNG seed")
    cli.add_argument('-cf', '--concentration-factor',
                     type=float,
                     nargs='*',
                     default=[1.], help="concentration factor for Dirichlet distribution. If only one value"
                                        "is passed, this is replicated K-times to match the param vector length")
    cli.add_argument('-e', '--eps-beta-params',
                     type=float,
                     nargs=2,
                     default=[5., 50.], metavar=("ALPHA", "BETA"), help="alpha and beta parameters for Beta distribution")
    cli.add_argument("-d", "--debug",
                     action="store_true",
                     help="additional inspection for debugging purposes")
    args = cli.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    # mkdir if it does not exist
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
        logging.info(f"{args.out_path} did not exist. Path created!")

    # simulate data and save it to file
    set_seed(args.seed)
    # preprocess args
    if len(args.concentration_factor) == 1:
        args.concentration_factor = args.concentration_factor[0]
    else:
        assert len(args.concentration_factor) == args.n_nodes

    data = simulate_full_dataset(
        Config(n_nodes=args.n_nodes, n_states=args.n_states, n_cells=args.n_cells, chain_length=args.chain_length),
        dir_alpha=args.concentration_factor,
        eps_a=args.eps_beta_params[0], eps_b=args.eps_beta_params[1])

    filename = f'simul_K{args.n_nodes}_A{args.n_states}_N{args.n_cells}_M{args.chain_length}'
    write_simulated_dataset_h5(data, args.out_path, filename, gt_mode='h5')
    logging.info(f'simulated dateset saved in {args.out_path}')
