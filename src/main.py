"""
Main CopyTree inference script.

Read input data (real/synthetic) and run VI inference algorithm.
Outputs K MAP trees, with cell assignments and copy number profiles for each clone.
"""
import argparse
import logging
import sys
import os

from inference.run import run
from utils.config import set_seed


def main(args):
    logging.info("running main program")
    copytree = run(args)
    logging.info("main is over")


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
    parser.add_argument("-i", "--input", dest="file_path",
                        type=validate_file, default='../obs_example.txt',
                        help="input data file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="out_path",
                        help="output data file", metavar="FILE")
    parser.add_argument("-d", "--debug", action="store_true", help="additional inspection for debugging purposes")
    parser.add_argument("--K", default=5, type=int, help="Number of nodes/clones")
    parser.add_argument("--A", default=7, type=int, help="Number of characters/copy number states")
    parser.add_argument("--L", default=10, type=int, help="Number of sampled arborescences")
    # parser.add_argument("--tmc-num-samples", default=10, type=int)
    args = parser.parse_args()
    # seed for reproducibility
    set_seed(args.seed)

    # logger setup
    logging.basicConfig(filename='out.log', level=logging.DEBUG if args.debug else logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    getattr(logging, args.log)

    main(args)
