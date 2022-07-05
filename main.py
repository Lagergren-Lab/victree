#!/usr/bin/env python3

import argparse
import logging

import networkx as nx
import numpy as np
import random
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from typing import Tuple

from matplotlib import pyplot as plt

from eps_utils import TreeHMM


def model_simple_markov(data, n_cells, n_sites, n_copy_states = 7) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    treeHMM = TreeHMM(n_copy_states, eps=0.3)
    cpd = treeHMM.cpd_table
    pair_cpd = treeHMM.cpd_pair_table
    # initialization
    C_r_m = 0
    C_u_m = 0

    # variables to store complete data in
    C_r = torch.zeros(n_sites, dtype=torch.int)
    C_u = torch.zeros(n_sites, dtype=torch.int)
    y_u = torch.zeros(n_sites, n_cells, dtype=torch.float)

    # simple transition matrix for root cn evolution
    pp = torch.eye(n_copy_states) * (1. - treeHMM.eps) + treeHMM.eps
    # no need for pyro.markov, range is equivalent
    # a simple for loop is used in the tutorials for sequential dependencies
    # ref: https://pyro.ai/examples/svi_part_ii.html#Sequential-plate
    for m in range(n_sites):

        # initial state case only depends on the parent initial state
        if m == 0:
            # starts with uniform over copy states
            dist_C_r_m = dist.Categorical(logits=torch.ones(n_copy_states))
            C_r_m = pyro.sample("C_r_{}".format(m), dist_C_r_m)
            dist_C_u_m = dist.Categorical(probs=pair_cpd[:, C_r_m])

        else:
            # save previous copy number
            C_r_m_1 = C_r_m
            # follow previous site's copy number for root node
            C_r_m = pyro.sample("C_r_{}".format(m), dist.Categorical(probs=pp[:, C_r_m]))

            # other states depend on 3 states
            dist_C_u_m = dist.Categorical(probs=cpd[:, C_u_m, C_r_m, C_r_m_1])

        C_u_m = pyro.sample("C_u_{}".format(m), dist_C_u_m)
        # save values in arrays
        C_r[m] = C_r_m
        C_u[m] = C_u_m
        y_u[m] = pyro.sample("y_u_{}".format(m), dist.Normal(C_u_m * torch.ones(n_cells), 1.0), obs=data[m])

    # debug
    # print(f"C_r_m {C_r}")
    # print(f"C_u_m {C_u}")
    # print(f"y_u_m {y_u}")

    return C_r, C_u, y_u


def model_tree_markov(data, n_cells, n_sites, n_copy_states, tree: nx.DiGraph):
    treeHMM = TreeHMM(n_copy_states, eps=0.3)
    cpd = treeHMM.cpd_table
    pair_cpd = treeHMM.cpd_pair_table

    n_nodes = len(tree.nodes)
    mu_n = 1.0  # TODO: change to random variable
    # initialization
    C_r_m = 0
    C_u_m = 0

    # variables to store complete data in
    C = torch.zeros(n_nodes, n_sites, dtype=torch.int)
    C_temp = {}
    y = torch.zeros(n_nodes, n_sites, n_cells, dtype=torch.float)

    # simple transition matrix for root cn evolution
    pp = torch.eye(n_copy_states) * (1. - treeHMM.eps) + treeHMM.eps

    for u in nx.dfs_preorder_nodes(tree):
        if u == 0:  # root case
            prior_tensor = torch.zeros(n_copy_states, dtype=torch.long)
            prior_tensor[2] = 1  # all probability on CN = 2
            C_r_m = pyro.sample("C_{}_{}".format(u, 0), dist.Categorical(prior_tensor))
            C_temp[u, 0] = C_r_m
            C[u, 0] = C_r_m
            y[u, 0] = pyro.sample("y_{}_{}".format(u, 0), dist.Normal(mu_n * C_r_m * torch.ones(n_cells), 1.0),
                                  obs=data[0])
        else:
            parent_idx = [pred for pred in tree.predecessors(u)][0]  # Raise exception if multiple pred found?
            C_p_0 = C_temp[parent_idx, 0]
            C_u_m = pyro.sample("C_{}_{}".format(u, 0), dist.Categorical(pp[:, C_p_0]))
            C_temp[u, 0] = C_u_m
            C[u, 0] = C_u_m
            y[u, 0] = pyro.sample("y_{}_{}".format(u, 0), dist.Normal(mu_n * C_u_m * torch.ones(n_cells), 1.0),
                                  obs=data[0])

        # no need for pyro.markov, range is equivalent
        # a simple for loop is used in the tutorials for sequential dependencies
        # ref: https://pyro.ai/examples/svi_part_ii.html#Sequential-plate
        for m in range(1, n_sites):

            # initial state case only depends on the parent initial state
            if u == 0:
                # save previous copy number
                C_r_m_1 = C_r_m if m != 0 else C_p_0
                # follow previous site's copy number for root node
                C_r_m = pyro.sample("C_{}_{}".format(u, m), dist.Categorical(probs=pp[:, C_r_m_1]))
                # save values in arrays
                C_temp[u, m] = C_r_m
                C[u, m] = C_r_m
                y[u, m] = pyro.sample("y_{}_{}".format(u, m), dist.Normal(C_r_m * torch.ones(n_cells), 1.0), obs=data[m])
            else:
                # save previous copy number
                parent_idx = [pred for pred in tree.predecessors(u)][0]  # Raise exception if multiple pred found
                C_p_m_1 = C_temp[parent_idx, m-1]
                C_p_m = C_temp[parent_idx, m]
                C_u_m_1 = C_temp[u, m-1]

                # other states depend on 3 states
                dist_C_u_m = dist.Categorical(probs=cpd[:, C_u_m_1, C_p_m, C_p_m_1])
                C_u_m = pyro.sample("C_{}_{}".format(u, m), dist_C_u_m)

                # save values in arrays
                C_temp[u, m] = C_u_m
                C[u, m] = C_u_m
                y[u, m] = pyro.sample("y_{}_{}".format(u, m), dist.Normal(C_u_m * torch.ones(n_cells), 1.0), obs=data[m])

    return C, y


def model_markov_tree_recursive(data, n_cells, n_sites, n_copy_states, tree: nx.DiGraph):
    """
    Recursive version of function above. Probably not needed. TODO: Delete when iterative version tested OK
    :param data:
    :param n_cells:
    :param n_sites:
    :param n_copy_states:
    :param tree:
    :return:
    """
    treeHMM = TreeHMM(n_copy_states, eps=0.3)
    cpd = treeHMM.cpd_table
    pair_cpd = treeHMM.cpd_pair_table

    n_nodes = len(tree.nodes)
    mu_n = 1.0  # TODO: change to random variable
    # initialization
    C_r_m = 0
    C_u_m = 0

    # variables to store complete data in
    C = torch.zeros(n_nodes, n_sites, dtype=torch.int)
    y = torch.zeros(n_nodes, n_sites, n_cells, dtype=torch.float)

    # simple transition matrix for root cn evolution
    pp = torch.eye(n_copy_states) * (1. - treeHMM.eps) + treeHMM.eps

    # priors
    prior_tensor = torch.zeros(n_copy_states, dtype=torch.long)
    prior_tensor[2] = 1  # all probability on CN = 2

    def markov_tree_recursion(u, m, C_p_m_1, C_p_m, C_u_m_1):
        if u == 0:
            if m == 0:
                C_r_0 = pyro.sample("C_{}_{}".format(0, 0), dist.Categorical(prior_tensor))
                C[u, m] = C_r_0
                #children_idx = [child for child in tree.successors(u)]
                for child in tree.successors(u):
                    markov_tree_recursion(u=child, m=m, C_p_m_1=None, C_p_m=C_r_0, C_u_m_1=None)
            else:
                # propagate m
                C_r_m = pyro.sample("C_{}_{}".format(u, m), dist.Categorical(probs=pp[:, C_u_m_1]))
                C[u, m] = C_r_m
                for child in tree.successors(u):
                    markov_tree_recursion(u=child, m=m, C_p_m_1=C_u_m_1, C_p_m=C_r_m, C_u_m_1=None)  # <--- How to access C_u_m_1 ???
        else:
            if m == 0:
                C_u_0 = pyro.sample("C_{}_{}".format(u, m), dist.Categorical(probs=pair_cpd[:, C_p_m]))
                C[u, m] = C_u_0
                for child in tree.successors(u):
                    markov_tree_recursion(u=child, m=m, C_p_m_1=None, C_p_m=C_u_0, C_u_m_1=None)

            else:
                C_u_m = pyro.sample("C_{}_{}".format(u, m), dist.Categorical(probs=cpd[:, C_p_m_1, C_p_m, C_u_m_1]))

                C[u, m] = C_u_m

        return 0

    # root and 0 site

    markov_tree_recursion(u=0, m=1, )



    for m in range(n_sites):

        # initial state case only depends on the parent initial state
        if m == 0:
            # starts with uniform over copy states
            dist_C_r_m = dist.Categorical(logits=torch.ones(n_copy_states))
            C_r_m = pyro.sample("C_r_{}".format(m), dist_C_r_m)
            dist_C_u_m = dist.Categorical(probs=pair_cpd[:, C_r_m] )


def main(args):
    if args.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # params
    n_cells = 3
    n_sites = 10
    n_copy_states = 5
    data = torch.ones(n_sites, n_cells)
    tree = nx.DiGraph()
    tree.add_edge(0, 1)
    tree.add_edge(0, 2)
    tree.add_edge(1, 3)
    tree.add_edge(1, 4)
    tree.add_edge(1, 5)
    nx.draw_networkx(tree)
    plt.show()
    graph = pyro.render_model(model_tree_markov, model_args=(data, n_cells, n_sites, n_copy_states, tree,))
    graph.render(outfile='./fig/graph.png')

    logging.info("Simulate data")
    # simulate latent variable as well as observations (synthetic data generation)
    # using "uncondition" handler
    unconditioned_model = poutine.uncondition(model_tree_markov)
    #C_r, C_u, y_u = unconditioned_model(data, n_cells, n_sites, n_copy_states, )
    C, y = unconditioned_model(data, n_cells, n_sites, n_copy_states, tree, )
    print(f"C_r: {C}")
    #print(f"C_u: {C_u}")
    print(f"y_u: {y}")


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Double chain HMM test"
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
