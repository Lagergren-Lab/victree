"""
Main CopyTree inference script.

Read input data (real/synthetic) and run VI inference algorithm.
Outputs K MAP trees, with cell assignments and copy number profiles for each clone.
"""
import argparse
import logging
import random
import sys

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
    try:
        run(args)
    except ValueError as ve:
        logging.error(f"program stopped with ValueError: {ve}")
    except Exception as e:
        logging.error(f"Unknown error: {e}")

    logging.debug("main is over")


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(
        description="CopyTree"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--n-iter", default=10, type=int)
    parser.add_argument("--log", default="DEBUG", action="store_true")
    # parser.add_argument("--tmc-num-samples", default=10, type=int)
    args = parser.parse_args()
    # seed for reproducibility
    set_seed(args.seed)

    # logger setup
    logging.basicConfig(filename='out.log', encoding='utf-8', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    getattr(logging, args.log)

    main(args)
