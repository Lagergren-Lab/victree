"""
Main CopyTree inference script.

Read input data (real/synthetic) and run VI inference algorithm.
Outputs K MAP trees, with cell assignments and copy number profiles for each clone.
"""
import argparse
import logging
import random
import sys
import os

import numpy as np
import torch

from inference.run import run

def set_seed(seed):
    # torch rng
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # python rng
    np.random.seed(seed)
    random.seed(seed)


def main(args):

    logging.debug("running main program")
    run(args)
    # try:
    #     run(args)
    # except ValueError as ve:
    #     logging.error(f"program stopped with ValueError: {ve}")
    # except Exception as e:
    #     logging.error(f"Unknown error: {e}")

    logging.debug("main is over")


def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(
        description="CopyTree"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--n-iter", default=10, type=int)
    parser.add_argument("--log", default="DEBUG", action="store_true")
    parser.add_argument("-i", "--input", dest="filename", 
                        required=True, type=validate_file,
                        help="input data file", metavar="FILE")
    # parser.add_argument("--tmc-num-samples", default=10, type=int)
    args = parser.parse_args()
    # seed for reproducibility
    set_seed(args.seed)

    # logger setup
    logging.basicConfig(filename='out.log', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    getattr(logging, args.log)

    main(args)
