#!/usr/bin/env python3

"""
Data simulation script.
"""

import argparse
import logging
import os.path

import h5py
import networkx as nx
import numpy as np
import torch
import torch.distributions as dist

from variational_distributions.joint_dists import VarTreeJointDist
from utils import tree_utils
from utils.config import Config, set_seed
from utils.eps_utils import h_eps, h_eps0
from utils.tree_utils import generate_fixed_tree
from variational_distributions.var_dists import qC, qZ, qEpsilonMulti, qMuTau, qPi, qT


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
        dictionary with keys: ['obs', 'raw', 'c', 'z', 'pi', 'mu', 'tau', 'eps', 'eps0', 'tree']
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


def simulate_quadruplet_data(M, A, tree: nx.DiGraph, eps_a, eps_b, eps_0, mu_v, mu_w, tau_v, tau_w):
    eps, c = simulate_copy_tree_data(4, M, A, tree, eps_a, eps_b, eps_0)
    z = [2, 3]
    mu = torch.tensor([mu_v, mu_w])
    tau = torch.tensor([tau_v, tau_w])
    y = simulate_observations_Normal(2, M, c, z, mu, tau)
    out_simul = {'obs': y,
                 'c': c,
                 'mu': mu,
                 'tau': tau,
                 'eps': eps,
                 'eps0': eps_0,
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


def simulate_observations_Normal(N, M, c, z, mu, tau):
    y = torch.zeros((M, N))
    for n in range(N):
        k = z[n]
        mu_n = c[k] * mu[n]
        y_n_dist = dist.Normal(mu_n, 1. / np.sqrt(tau[n]))
        y[:, n] = y_n_dist.sample()

    return y


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
    # FIXME: documentation
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
                     metavar="DELTA",
                     default=[10.], help="concentration factor for Dirichlet distribution. If only one value"
                                         "is passed, this is replicated K-times to match the param vector length")
    cli.add_argument('-e', '--eps-params',
                     type=float,
                     nargs=2,
                     default=[1., 50.], metavar=("ALPHA", "BETA"),
                     help="alpha and beta parameters for Beta distribution")
    cli.add_argument("--mutau-params", default=[1., 10., 500., 50.], nargs=4, type=float,
                     help="prior on mu-tau (Normal-Gamma dist)",
                     metavar=("NU", "LAMBDA", "ALPHA", "BETA"))
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
    conc_fact_str = "-".join(map(str, map(int, args.concentration_factor)))
    if len(args.concentration_factor) == 1:
        args.concentration_factor = args.concentration_factor[0]
        conc_fact_str = str(int(args.concentration_factor))
    else:
        assert len(args.concentration_factor) == args.n_nodes

    data = simulate_full_dataset(
        Config(n_nodes=args.n_nodes, n_states=args.n_states, n_cells=args.n_cells, chain_length=args.chain_length),
        dir_alpha=args.concentration_factor,
        mu0=args.mutau_params[0], lambda0=args.mutau_params[1], alpha0=args.mutau_params[2], beta0=args.mutau_params[3],
        eps_a=args.eps_params[0], eps_b=args.eps_params[1])

    filename = f'simul_' \
               f'k{args.n_nodes}a{args.n_states}n{args.n_cells}m{args.chain_length}' \
               f'e{int(args.eps_params[0])}-{int(args.eps_params[1])}' \
               f'd{conc_fact_str}' \
               f'mt{"-".join(map(str, map(int, args.mutau_params)))}'
    write_simulated_dataset_h5(data, args.out_path, filename, gt_mode='h5')
    logging.info(f'simulated dateset saved in {args.out_path}')
