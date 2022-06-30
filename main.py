#!/usr/bin/env python3

import argparse
import torch
import pyro
import pyro.distributions as dist
import logging
import pyro.poutine as poutine


def model_simple_markov(data, n_cells, n_sites, n_copy_states = 7) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # initialization
    C_r_m = 0
    C_u_m = 0

    # variables to store complete data in
    C_r = torch.zeros(n_sites, )
    C_u = torch.zeros(n_sites, )
    y_u = torch.zeros(n_sites, n_cells)

    a = 2
    # no need for pyro.markov, range is equivalent
    # a simple for loop is used in the tutorials for sequential dependencies
    # ref: https://pyro.ai/examples/svi_part_ii.html#Sequential-plate
    for m in range(n_sites):
        dist_C_r_m = dist.Binomial(C_r_m + a, torch.tensor(1.0))
        # save previous copy number
        C_r_m_1 = C_r_m
        C_r_m = pyro.sample("C_r_{}".format(m), dist_C_r_m)

        # initial state case only depends on the parent initial state
        if m == 0:
            dist_C_u_m = dist.Binomial(C_r_m, torch.tensor(1.0))
        # other states depend on 3 states
        else:
            dist_C_u_m = dist.Binomial(C_r_m + C_r_m_1 + C_u_m, torch.tensor(1.0))

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
    n_sites = 5
    n_copy_states = 3
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
    parser = argparse.ArgumentParser(
        description="MAP Baum-Welch learning Bach Chorales"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--tmc-num-samples", default=10, type=int)
    args = parser.parse_args()
    main(args)
