#!/usr/bin/env python3

"""
Program Main
- read and parse input arguments
- setup logger
- run algorithm
"""
import argparse
import logging
import math
import os
import pathlib
import sys
import time

from inference.run import run
from utils.config import set_seed


def main(args):
    logging.info("running main program")
    start = time.time()
    try:
        run(args)
    except Exception as e:
        logging.exception("main fail traceback")
    # TODO: save time in output
    tot_time = time.time() - start
    logging.info(f"main is over. Total exec time: {tot_time // 60}m {math.ceil(tot_time % 60)}s")


def set_logger(debug: bool, out_dir: str | None = None):
    level = logging.DEBUG if debug else logging.INFO
    c_handler = logging.StreamHandler(sys.stdout)

    c_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger = logging.root
    logger.setLevel(level)
    logger.addHandler(c_handler)

    if out_dir is not None:
        f_handler = logging.FileHandler(os.path.join(out_dir, "victree.log"))
        f_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%y%m%d-%H:%M:%S'))
        logger.addHandler(f_handler)


def validate_path(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f


def validate_args(args):
    if not args.sieving[0] > 1:
        args.sieving[1] = 0
    elif args.sieving[1] < 2:
        raise argparse.ArgumentError(args.sieving, message=f"If sieving, num of sieving iterations must be > 1")

    if len(args.prior_pi) == 1:
        args.prior_pi = args.prior_pi[0]
    elif len(args.prior_pi) != args.n_nodes:
        raise argparse.ArgumentError(args.prior_pi, message=f"Prior for pi must be either length 1 or K. "
                                                            f"K was set to {args.n_nodes}, but pi prior "
                                                            f"has length {len(args.prior_pi)}")
    if args.z_init not in ["random", "uniform", "kmeans", "gmm"]:
        raise argparse.ArgumentError(args.z_init, message=f"Init method {args.z_init} for"
                                                          f"qZ variational distribution is not valid. Check usage.")


def none_or_str(value):
    if value == 'None':
        return None
    return value


def parse_args(parser):
    args = parser.parse_args()
    # if args.config_file:
    #     data = yaml.load(args.config_file)
    #     delattr(args, 'config_file')
    # TODO: continue implementation for yaml config with cli args
    #   check this: https://codereview.stackexchange.com/questions/79008/parse-a-config-file-and-add-to-command-line-arguments-using-argparse-in-python
    return args


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        description="VIC-Tree, variational inference on clonal tree with single-cell DNA data"
    )
    # parser.add_argument("-c", "--config-file", dest='config_file', type=argparse.FileType(mode='r'))
    parser.add_argument("-i", "--input", dest="file_path",
                        type=validate_path, default='../datasets/simul_k5a7n300m1000e1-50d10mt1-10-500-50.h5',
                        help="input data file", metavar="FILE")
    parser.add_argument("--tree", dest="tree_path",
                        type=str, default="",
                        help="newick tree file for fixed tree inference", metavar="FILE")
    parser.add_argument("-o", "--output", dest="out_dir",
                        help="output dir", metavar="DIR", default="./output")
    parser.add_argument("-d", "--debug", action="store_true", help="additional inspection for debugging purposes")
    parser.add_argument("-s", "--seed", default=42, type=int, help="RNG seed")
    parser.add_argument("-n", "--n-iter", default=20, type=int, help="VI iterations")
    parser.add_argument("-a", "--diagnostics", action="store_true", help="store data of var dists during optimization")
    parser.add_argument("-K", "--n-nodes", default=5, type=int, help="number of nodes/clones")
    parser.add_argument("-A", "--n-states", default=7, type=int, help="number of characters/copy number states")
    parser.add_argument("-S", "--step-size", default=.1, type=float, help="step-size for partial updates")
    parser.add_argument("-L", "--tree-sample-size", default=10, type=int, help="number of sampled arborescences")
    parser.add_argument("--r-tol", default=10e-4, type=float, help="relative tolerance for early stopping")

    # priors parameters
    parser.add_argument("--prior-eps", default=[1., 50.], nargs=2, type=float, help="prior on epsilon  (Beta dist)",
                        metavar=("ALPHA", "BETA"))
    parser.add_argument("--prior-mutau", default=[1., 100., 5000., 500.], nargs=4, type=float,
                        help="prior on mu-tau (Normal-Gamma dist)",
                        metavar=("NU", "LAMBDA", "ALPHA", "BETA"))
    parser.add_argument("--prior-pi", default=[10.], nargs='*', type=float,
                        help="prior on pi  (Dirichlet dist). If uniform, one single value can be specified,"
                             "otherwise provide as many values as the specified K parameter (number of nodes)",
                        metavar="DELTA")
    # initialization
    parser.add_argument("--z-init", default="random", type=str, metavar="TYPE_STR",
                        help="initialization of clustering variable qZ, can be [`random`, `uniform`, `kmeans`], "
                             "default to `random`")
    parser.add_argument("--eps-init", default="random", type=str, metavar="TYPE_STR",
                        help="initialization of alpha/beta params in qEpsilon, can be [`random`, `uniform`,"
                             " `non-mutation`, `prior`, `data`], default to `random`")
    parser.add_argument("--mt-init", default="data", type=str, metavar="TYPE_STR",
                        help="initialization of (nu, lambda, alpha, beta) params qMuTau, can be [`data`, `data-size`,"
                             " `prior`], default to `data`")
    parser.add_argument("--c-init", default="diploid", type=str, metavar="TYPE_STR",
                        help="initialization of copy number variable qC, can be [`clonal`, `uniform`,"
                             " `diploid`, `random`], default to `diploid`")
    # Meta algorithms
    parser.add_argument("--split", default="mixed", type=none_or_str, metavar="TYPE_STR",
                        help="specify which split algorithm to run [`naive`, `categorical`, `mixed`, `inlier`, `mixed`, `None`],"
                             " defaults to `mixed`")
    parser.add_argument("--merge_and_split_interval", default=5, type=int,
                        help="specify the number of iterations between each call to the merge and split algorithm.")
    parser.add_argument("--SVI", action="store_true", help="Run Stochastic Variational inference.")
    parser.add_argument("--sieving", default=[1, 0], nargs=2, type=int, help="number of sieving runs prior to start",
                        metavar=("N_RUNS", "N_ITER"))
    parser.add_argument("--impute-nans", default="ignore", type=str, metavar="TYPE_STR",
                        help="specify how to deal with NaNs in the data, can be [`ignore`, `remove`],"
                             " defaults to `ignore`")
    parser.add_argument("--qT-temp", default=1., type=float, help="Anneal the qT distribution from qT-temp to 1.")
    parser.add_argument("--qZ-temp", default=1., type=float, help="Anneal the qZ distribution from qZ-temp to 1.")

    # parser.add_argument("--tmc-num-samples", default=10, type=int)
    args = parser.parse_args()
    validate_args(args)  # custom validation on args
    # seed for reproducibility
    set_seed(args.seed)

    # create the output path if it does not exist
    if not os.path.exists(args.out_dir):
        pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=False)

    # logger setup
    set_logger(args.debug, args.out_dir)

    # main program
    main(args)
