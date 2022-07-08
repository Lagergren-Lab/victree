#!/usr/bin/env python3

import argparse
import logging
import random

import networkx as nx
from networkx.algorithms.tree.coding import NotATree
from networkx.algorithms.tree.recognition import is_arborescence
import numpy as np
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from matplotlib import pyplot as plt
from eps_utils import TreeHMM


def model_tree_markov(data, n_cells, n_sites, n_copy_states, tree: nx.DiGraph):
    # check that tree is with maximum in-degree =1
    if not is_arborescence(tree):
        raise NotATree("The provided graph is not a tree/arborescence")


    n_nodes = len(tree.nodes)

    # PRIORS PARAMETERS
    # normal params
    # location and scale
    mu_0 = torch.tensor(0.)
    sigma_0 = torch.tensor(1.)

    # inverse gamma params
    # concentration and rate
    a0, b0 = torch.tensor(.001), torch.tensor(.001)

    # beta params
    # shape alpha, shape beta
    alpha0, beta0 = torch.tensor(10.), torch.tensor(40.)

    mu = pyro.sample("mu", dist.Normal(mu_0, sigma_0).expand([n_cells]))
    sigma2 = pyro.sample("sigma", dist.InverseGamma(a0, b0))
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
                C_p_m = C_dict[p, m]

                zipping_dist = None
                if m == 0:
                    # initial state case only depends on the parent initial state
                    # use pair_cpd as m-1 is not available
                    zipping_dist = dist.Categorical(pair_cpd[:, C_p_m])
                else:
                    # previous copy numbers
                    C_p_m_1 = C_dict[p, m - 1]
                    C_u_m_1 = C_dict[u, m - 1]

                    # other states depend on 3 states
                    zipping_dist = dist.Categorical(probs=cpd[:, C_u_m_1, C_p_m, C_p_m_1])

                C_u_m = pyro.sample("C_{}_{}".format(u, m), zipping_dist)

                # save values in dict
                C_dict[u, m] = C_u_m
                y[m, z == u] = pyro.sample("y_{}_{}".format(u, m), dist.Normal(mu[z == u] * C_u_m, sigma2.sqrt()),
                                      obs=data[m, z == u])

    C = torch.empty((n_nodes, n_sites))
    for u, m in C_dict.keys():
        C[u, m] = C_dict[u,m]

    return C, y, z


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
    f.savefig("./fig/tree_topology.png")

    # draw bayes net
    graph = pyro.render_model(model_tree_markov, model_args=(data, n_cells, n_sites, n_copy_states, tree,))
    graph.render(outfile='./fig/graph.png')

    logging.info("Simulate data")
    # simulate latent variable as well as observations (synthetic data generation)
    # using "uncondition" handler
    unconditioned_model = poutine.uncondition(model_tree_markov)
    #C_r, C_u, y_u = unconditioned_model(data, n_cells, n_sites, n_copy_states, )
    C, y, z = unconditioned_model(data, n_cells, n_sites, n_copy_states, tree, )
    print(f"C: {C}")
    print(f"y: {y}")
    print(f"z: {z}")


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Tree HMM test"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cuda", action="store_true")
    # parser.add_argument("--tmc-num-samples", default=10, type=int)
    args = parser.parse_args()
    # seed for reproducibility
    # torch rng
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # python rng
    np.random.seed(args.seed)
    random.seed(args.seed)

    main(args)
