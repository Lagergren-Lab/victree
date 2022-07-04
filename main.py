#!/usr/bin/env python3

import argparse
import logging
import numpy as np
import random
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from eps_utils import TreeHMM

def model_simple_markov(data, n_cells, n_sites, n_copy_states = 7) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            dist_C_u_m = dist.Categorical(probs=pair_cpd[:, C_r_m] )

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


def main(args):
    if args.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # params
    n_cells = 3
    n_sites = 10
    n_copy_states = 5
    data = torch.ones(n_sites, n_cells)
    graph = pyro.render_model(model_simple_markov, model_args=(data, n_cells, n_sites, n_copy_states, ))
    graph.render(outfile='./fig/graph.png')

    logging.info("Simulate data")
    # simulate latent variable as well as observations (synthetic data generation)
    # using "uncondition" handler
    unconditioned_model = poutine.uncondition(model_simple_markov)
    C_r, C_u, y_u = unconditioned_model(data, n_cells, n_sites, n_copy_states, )
    print(f"C_r: {C_r}")
    print(f"C_u: {C_u}")
    print(f"y_u: {y_u}")


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
