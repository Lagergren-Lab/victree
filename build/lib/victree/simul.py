#!/usr/bin/env python3

"""
Data simulation script.
"""

import argparse
import logging
import math
import os.path
from pathlib import Path

import anndata
import h5py
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.distributions as dist
from scgenome.tools import create_bins

from utils.tree_utils import tree_to_newick
from variational_distributions.joint_dists import VarTreeJointDist
from utils import tree_utils
from utils.config import Config, set_seed
from utils.eps_utils import h_eps, h_eps0
from variational_distributions.var_dists import qC, qZ, qEpsilonMulti, qMuTau, qPi, qT, qCMultiChrom


def simulate_full_dataset(config: Config, eps_a=500., eps_b=50000., mu0=1., lambda0=1000.,
                          alpha0=500., beta0=50., dir_delta: [float | list[float]] = 1., tree=None, raw_reads=True,
                          chr_df: pd.DataFrame | None = None, nans: bool = False,
                          fixed_z:torch.Tensor = None, cne_length_factor: int = 0):
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
        chr_df: pd.DataFrame with cols ['chr', 'start', 'end']
            If None, a real-data like 24 chr binning of the genome is used
        cne_length_factor: int, the higher, the longer the copy number events will be on average.
         if set to 0, no correction is performed and copy number transition is only
         determined by the markov model. A copy number event, once started, will be terminated
         with probability 1 / cne_length_factor along the chain.

    Returns:
        dictionary with keys: ['obs', 'raw', 'c', 'z', 'pi', 'mu', 'tau', 'eps', 'eps0', 'tree', 'chr_idx', 'adata']
    """
    # set chr_idx
    if chr_df is None:
        chr_idx = []
    else:
        # get idx where chr changes and remove the leading 0
        sorted_df = chr_df.sort_values(['chr', 'start']).reset_index()
        chr_idx = sorted_df.index[sorted_df['chr'].ne(sorted_df['chr'].shift())].to_list()[1:]
        if chr_df.shape[0] != config.chain_length:
            logging.debug(f"due to chromosome splitting, the total number of sites changed:"
                          f" {config.chain_length} -> {chr_df.shape[0]}")
            config.chain_length = chr_df.shape[0]
    config.chromosome_indexes = chr_idx
    n_chromosomes = config.n_chromosomes
    ext_chr_idx = [0] + chr_idx + [config.chain_length]

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
        h_eps_uv = h_eps(config.n_states, eps[u, v].item())

        for ci in range(n_chromosomes):
            lb, ub = ext_chr_idx[ci], ext_chr_idx[ci + 1]
            c[v, lb] = torch.distributions.Categorical(probs=t0).sample()
            basal_cn_idx = lb

            on_cne = False
            for m in range(lb + 1, ub):
                # j', j, i', i

                if on_cne and torch.rand(1) < 1 / cne_length_factor:
                    # terminate cn event
                    c[v, m] = c[v, basal_cn_idx]
                else:
                    transition = h_eps_uv[:, c[v, m - 1], c[u, m], c[u, m - 1]]
                    c[v, m] = torch.distributions.Categorical(probs=transition).sample()

                # in case of copy number change, keep track of previous cn
                # and alter transition to favor copy number comeback
                if cne_length_factor != 0 and c[v, m] != c[v, m-1]:
                    # set flag to true when cn event starts, back to false if it ends
                    on_cne = c[v, m] != c[v, basal_cn_idx]

    # sample mu_n, tau_n
    tau = torch.distributions.Gamma(alpha0, beta0).sample((config.n_cells,))
    mu = torch.distributions.Normal(mu0, 1. / torch.sqrt(lambda0 * tau)).sample()
    assert mu.shape == tau.shape
    # sample assignments
    if isinstance(dir_delta, float):
        dir_alpha_tensor = torch.ones(config.n_nodes) * dir_delta
    elif isinstance(dir_delta, list):
        dir_alpha_tensor = torch.tensor(dir_delta)
    else:
        raise ValueError(f"dir_alpha param must be either a k-size list of float or a float (not {type(dir_delta)})")
    pi = torch.distributions.Dirichlet(dir_alpha_tensor).sample()
    z = torch.distributions.Categorical(pi).sample((config.n_cells,)) if fixed_z is None else fixed_z
    # sample observations
    obs_mean = c[z, :] * mu[:, None]  # n_cells x chain_length
    scale_expanded = torch.pow(tau, -1 / 2).reshape(-1, 1).expand(-1, config.chain_length)
    # (chain_length x n_cells)
    obs = torch.distributions.Normal(obs_mean, scale_expanded).sample().clamp(min=0.)
    obs = obs.T

    raw_counts = sample_raw_counts_from_corrected_data(obs) if raw_reads is True else None

    if nans:
        # 8% of sites is nan
        nan_idx = torch.rand(config.chain_length) < .08
        # make sure each chr starts with a non-missing datum
        nan_idx[[0] + chr_idx] = False
        obs[nan_idx, :] = torch.nan
    assert obs.shape == (config.chain_length, config.n_cells)

    # handy anndata object
    adata = make_anndata(obs, raw_counts, chr_df, c, z, mu, tree)

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
        'tree': tree,
        'chr_idx': chr_idx,
        'adata': adata
    }
    return out_simul


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


def generate_chromosome_binning(n: int, method: str = 'real', n_chr: int | None = None) -> pd.DataFrame:
    """
    Parameters
    ----------
    n: int, total number of bins/sites. Note: given the specified number of chromosomes, the number of bins in the
        dataframe might be slightly larger
    method: str, whether to use the human genome 19 chromosomes reference or a uniformly distributed set of chromosomes;
        can be 'real' or 'uniform'
    n_chr: int, number of chromosomes in the 'uniform' method

    Returns
    -------
    dataframe with start, end and chr columns, sorted by chr,start
    """
    if method == 'real':
        ord_chr = [str(c) for c in range(1, 23)] + ['X', 'Y']
        # https://www.ncbi.nlm.nih.gov/grc/human/data
        hg19_total_length = 3099734149
        binsize = math.ceil(hg19_total_length / n)
        splits_df = create_bins(binsize)
        splits_df.chr = pd.Categorical(splits_df.chr, categories=ord_chr, ordered=True)

    elif method == 'uniform':
        binsize = 1000
        if n_chr is None:
            raise ValueError("Must provide number of chromosomes for `uniform` chromosome splits")
        chr_width = n // n_chr
        chr = pd.Categorical([str(c) for c in range(1, n_chr + 1)], ordered=True)
        pos = pd.DataFrame({
            'start': [s * binsize + 1 for s in range(chr_width)],
            'end': [(s + 1) * binsize for s in range(chr_width)]
        })
        splits_df = pos.merge(pd.DataFrame({'chr': chr}), how='cross')
    else:
        raise NotImplementedError(f"Method {method} for chromosome splits creation is not available.")

    return splits_df


def make_anndata(obs, raw_counts, chr_dataframe, c, z, mu, tree, obs_names: list | None = None):

    adata = anndata.AnnData(raw_counts.T.numpy())
    adata.layers['copy'] = obs.T.numpy()

    cn_state = c[z, :].numpy()
    assert cn_state.shape == adata.shape
    adata.layers['state'] = cn_state
    adata.obs['clone'] = z.numpy()
    adata.obs['clone'] = adata.obs['clone'].astype('category')
    adata.obs['baseline'] = mu.numpy()
    adata.uns['tree-newick'] = np.array([tree_to_newick(tree)], dtype='S')

    if chr_dataframe is None:
        chr_dataframe = generate_chromosome_binning(adata.n_vars, method='uniform', n_chr=1)
        chr_dataframe = chr_dataframe[:adata.n_vars]
    adata.var = chr_dataframe

    if obs_names is None:
        pad_width = math.ceil(math.log10(adata.n_obs))
        obs_names = ['c' + str(i).zfill(pad_width) for i in range(adata.n_obs)]
    adata.obs_names = pd.Series(obs_names)

    return adata


def simulate_quadruplet_data(M, A, tree: nx.DiGraph, eps_a, eps_b, eps_0):
    eps, c = simulate_copy_tree_data(4, M, A, tree, eps_a, eps_b, eps_0)
    z = [2, 3]
    mu = torch.tensor([0.8, 1.3])
    tau = torch.tensor([10., 10.])
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


def generate_dataset_var_tree(config: Config,
                              nu_prior=1., lambda_prior=100.,
                              alpha_prior=500., beta_prior=50.,
                              dir_alpha=1., eps_a=150., eps_b=4500., chrom: str | int = 1,
                              ret_anndata=False, cne_length_factor: int = 0):
    # set up default with one chromosome
    chr_df = None
    if chrom == 'real':
        chr_df = generate_chromosome_binning(config.chain_length)
    elif isinstance(chrom, int):
        if chrom != 1:
            chr_df = generate_chromosome_binning(config.chain_length, method='uniform', n_chr=chrom)
    else:
        raise ValueError(f"chrom argument `{chrom}` does not match any available option")

    simul_data = simulate_full_dataset(config, eps_a=eps_a, eps_b=eps_b, mu0=nu_prior, lambda0=lambda_prior,
                                       alpha0=alpha_prior, beta0=beta_prior, dir_delta=dir_alpha, chr_df=chr_df,
                                       cne_length_factor=cne_length_factor)

    if chrom != 1:
        fix_qc = qCMultiChrom(config, true_params={
            "c": simul_data['c']
        })
    else:
        # TODO: remove and leave just qCMultiChrom with one chrom when completely implemented
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
    if ret_anndata:
        return joint_q, simul_data['adata']
    else:
        return joint_q

def main():
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
    cli.add_argument('--mutau-params', default=[1., 10., 500., 50.], nargs=4, type=float,
                     help="prior on mu-tau (Normal-Gamma dist)",
                     metavar=("NU", "LAMBDA", "ALPHA", "BETA"))
    cli.add_argument('-d', '--debug',
                     action="store_true",
                     help="additional inspection for debugging purposes")
    cli.add_argument('--n-chromosomes',
                     type=str,
                     default='1', help="number of chromosomes i.e. separate copy number chains")
    cli.add_argument('--nans',
                     action="store_true",
                     help="set 1%% of the total sites to nan")
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

    if args.n_chromosomes == 'real':
        chr_df = generate_chromosome_binning(args.chain_length, method='real')
    else:
        try:
            n_chr = int(args.n_chromosomes)
            chr_df = generate_chromosome_binning(args.chain_length, method='uniform', n_chr=n_chr)
        except ValueError as v:
            raise argparse.ArgumentTypeError(f"wrong number of chromosomes param: {args.n_chromosomes}")

    config = Config(n_nodes=args.n_nodes, n_states=args.n_states, n_cells=args.n_cells, chain_length=args.chain_length)
    data = simulate_full_dataset(config, eps_a=args.eps_params[0], eps_b=args.eps_params[1], mu0=args.mutau_params[0],
                                 lambda0=args.mutau_params[1], alpha0=args.mutau_params[2], beta0=args.mutau_params[3],
                                 dir_delta=args.concentration_factor, chr_df=chr_df, nans=args.nans)
    filename = f'simul_' \
               f'k{config.n_nodes}a{config.n_states}n{config.n_cells}m{config.chain_length}' \
               f'e{int(args.eps_params[0])}-{int(args.eps_params[1])}' \
               f'd{conc_fact_str}' \
               f'mt{"-".join(map(str, map(int, args.mutau_params)))}'
    data['adata'].write_h5ad(Path(args.out_path, filename + '.h5ad'))
    # write_simulated_dataset_h5(data, args.out_path, filename, gt_mode='h5')
    logging.info(f'simulated dateset saved in {args.out_path}')


# script for simulating data
if __name__ == '__main__':
    main()
