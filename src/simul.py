"""
Data simulation script.
Uses Pyro-PPL for modelling the data only with data generation purposes.
"""
# FIXME: data simulation is not working, need to fix pair_cpd table,
#   maybe C_dict needs to store plain python variables and not pyro variables

import argparse
import logging
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

from inference.copy_tree import JointVarDist
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
        mu = pyro.sample("mu_{}".format(n), dist.Normal(mu_0, 1. / (lambda_0 * tau.sqrt())))
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
                    y[m, n] = pyro.sample("y_{}_{}".format(m, n), dist.Normal(mu[n] * C_r_m, 1.0 / (lambda_0 * tau[n].sqrt())), obs=data[m, n])
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
                y[m, ind] = pyro.sample("y_{}_{}".format(u, ind), dist.Normal(mu[ind] * C_u_m, 1.0 / (lambda_0 * tau[ind].sqrt())), obs=data[m, ind])

    C = torch.empty((n_nodes, n_sites))
    for u, m in C_dict.keys():
        C[u, m] = C_dict[u, m]

    return C, y, z, pi, mu, tau, eps


def simulate_full_dataset(config: Config, eps_a=1., eps_b=3., mu0=1., lambda0=10.,
                          alpha0=500., beta0=50.):
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
    tree = nx.random_tree(config.n_nodes, create_using=nx.DiGraph)
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
            transition = h_eps_uv[:, c[v, m-1], c[u, m], c[u, m-1]]
            c[v, m] = torch.distributions.Categorical(probs=transition).sample()

    # sample mu_n, tau_n
    tau = torch.distributions.Gamma(alpha0, beta0).sample((config.n_cells,))
    mu = torch.distributions.Normal(mu0, 1./torch.sqrt(lambda0 * tau)).sample()
    assert mu.shape == tau.shape
    # sample assignments
    pi = torch.distributions.Dirichlet(torch.ones(config.n_nodes)).sample()
    z = torch.distributions.Categorical(pi).sample((config.n_cells,))
    # sample observations
    obs_mean = c[z, :] * mu[:, None]  # n_cells x chain_length
    scale_expanded = torch.pow(tau, -2).reshape(-1, 1).expand(-1, config.chain_length)
    # (chain_length x n_cells)
    obs = torch.distributions.Normal(obs_mean, scale_expanded).sample()
    obs = obs.T
    assert obs.shape == (config.chain_length, config.n_cells)
    out_simul = {
        'obs': obs,
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
                    y[m, n] = pyro.sample("y_{}_{}".format(m, n), dist.Normal(mu[n] * C_r_m, 1.0 / tau[n].sqrt()), obs=data[m, n])
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
                y[m, ind] = pyro.sample("y_{}_{}".format(u, ind), dist.Normal(mu[ind] * C_u_m, 1.0 / (tau[ind].sqrt())), obs=data[m, ind])

    C = torch.empty((n_nodes, n_sites))
    for u, m in C_dict.keys():
        C[u, m] = C_dict[u, m]

    return C, y, z, pi, mu, tau, eps


def write_simulated_dataset_h5(dest_path, data):
    f = h5py.File(dest_path, 'w')
    x_ds = f.create_dataset('X', data=data['obs'])
    layers_grp = f.create_group('layers')
    z = data['z']
    cn_state = data['c'][z, :].T
    assert(cn_state.shape == x_ds.shape)
    layers_grp.create_dataset('state', data=cn_state)

    gt_group = f.create_group('gt')
    gt_group.create_dataset('cell_assignment', data=data['z'])
    gt_group.create_dataset('tree', data=torch.tensor(list(data['tree'].edges.data('weight'))))
    # TODO: write all remaining data 'eps, mu, tau, ...'

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
    assert(cn_state.shape == x_ds.shape)
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
    print(f"tree: {tree_to_newick(tree, 0)}")
    print(f"C: {C}")
    print(f"y: {y}")
    print(f"z: {z}")


def tree_to_newick(g: nx.DiGraph, root=None):
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
            subgs.append(tree_to_newick(g, root=child))
        else:
            subgs.append(str(child))
    return "(" + ','.join(subgs) + ")" + str(root)


if __name__ == '__main__':
    # save data to h5 file
    #write_sample_dataset_h5('../data_example.h5')

    # simulate data and save it to file
    set_seed(42)
    data = simulate_full_dataset(Config(n_nodes=5, n_states=7, n_cells=300, chain_length=1000))
    write_simulated_dataset_h5('../datasets/n5_c300_l1k.h5', data)

    ## parse arguments
    #parser = argparse.ArgumentParser(
    #    description="Tree HMM test"
    #)
    #parser.add_argument("--seed", default=42, type=int)
    #parser.add_argument("--cuda", action="store_true")
    ## parser.add_argument("--tmc-num-samples", default=10, type=int)
    #args = parser.parse_args()
    ## seed for reproducibility
    ## torch rng
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed(args.seed)
    #torch.cuda.manual_seed_all(args.seed)
    ## python rng
    #np.random.seed(args.seed)
    #random.seed(args.seed)

    #main(args)


def generate_dataset_var_tree(config: Config) -> JointVarDist:
    simul_data = simulate_full_dataset(config, eps_a=2, eps_b=5)

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
    })

    fix_qpi = qPi(config, true_params={
        "pi": simul_data['pi']
    })

    fix_qt = qT(config, true_params={
        "tree": simul_data['tree']
    })

    joint_q = JointVarDist(config, simul_data['obs'], fix_qc, fix_qz, fix_qt, fix_qeps, fix_qmt, fix_qpi)
    return joint_q