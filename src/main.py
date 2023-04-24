#!/usr/bin/env python3

"""
Main CopyTree inference script.

Read input data (real/synthetic) and run VI inference algorithm.
Outputs K MAP trees, with cell assignments and copy number profiles for each clone.
"""
import argparse
import logging
import math
import sys
import os
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
    tot_time = time.time() - start
    logging.info(f"main is over. Total exec time: {tot_time // 60}m {math.ceil(tot_time % 60)}s")


def set_logger(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    f_handler = logging.FileHandler('out.log')
    c_handler = logging.StreamHandler(sys.stdout)

    f_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%y%m%d-%H:%M:%S'))
    c_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger = logging.root
    logger.setLevel(level)
    logger.addHandler(f_handler)
    logger.addHandler(c_handler)

def validate_path(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f


def validate_args(args):
    if not args.sieving[0] > 1:
        args.sieving[1] = 0


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        description="VIC-Tree, variational inference on clonal tree with single-cell DNA data"
    )
    # parser.add_argument("-l", "--log-level", default="DEBUG", action="store_true")
    parser.add_argument("-i", "--input", dest="file_path",
                        type=validate_path, default='./datasets/n5_c300_l1k.h5',
                        help="input data file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="out_dir",
                        help="output dir", metavar="DIR")
    parser.add_argument("-d", "--debug", action="store_true", help="additional inspection for debugging purposes")
    parser.add_argument("-s", "--seed", default=42, type=int, help="RNG seed")
    parser.add_argument("-n", "--n-iter", default=10, type=int, help="VI iterations")
    parser.add_argument("-a", "--diagnostics", action="store_true", help="store data of var dists during optimization")
    parser.add_argument("-K", "--n-nodes", default=5, type=int, help="number of nodes/clones")
    parser.add_argument("-A", "--n-states", default=7, type=int, help="number of characters/copy number states")
    parser.add_argument("-S", "--step-size", default=.1, type=float, help="step-size for partial updates")
    parser.add_argument("-L", "--tree-sample-size", default=10, type=int, help="number of sampled arborescences")
    parser.add_argument("--sieving", default=[1, 0], nargs=2, type=int, help="number of sieving runs prior to start",
                        metavar=("N_RUNS", "N_ITER"))
    # parser.add_argument("--tmc-num-samples", default=10, type=int)
    args = parser.parse_args()
    validate_args(args)  # custom validation on args
    # seed for reproducibility
    set_seed(args.seed)

    # logger setup
    set_logger(args.debug)

    # main program
    main(args)
