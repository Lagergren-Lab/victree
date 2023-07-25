import argparse
import os

from utils.config import set_seed
from utils.data_handling import DataHandler

def validate_path(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f


def run_analysis(args):
    """
    Instantiate configuration object, variational distributions
    and observations. Run main inference algorithm.
    """
    # ---
    # Import data
    # ---
    if args.import_data:
        data_handler = DataHandler(args.file_path)
        obs = data_handler.norm_reads

    # ---
    # Create configuration object
    # ---


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        description="VIC-Tree analysis")
    parser.add_argument("-i", "--input", dest="file_path",
                        type=validate_path, default='../output/simul_k5a7n300m1000e1-50d10mt1-10-500-50.h5',
                        help="output model file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="out_dir",
                        help="output dir", metavar="DIR", default="./output")
    parser.add_argument("-s", "--seed", default=42, type=int, help="RNG seed")
    parser.add_argument("-n", "--n-iter", default=20, type=int, help="VI iterations")
    parser.add_argument("-a", "--diagnostics", action="store_true", help="store data of var dists during optimization")
    parser.add_argument("-K", "--n-nodes", default=5, type=int, help="number of nodes/clones")
    parser.add_argument("-A", "--n-states", default=7, type=int, help="number of characters/copy number states")
    parser.add_argument("-S", "--step-size", default=.1, type=float, help="step-size for partial updates")
    parser.add_argument("-L", "--tree-sample-size", default=10, type=int, help="number of sampled arborescences")
    parser.add_argument("--r-tol", default=10e-4, type=float, help="relative tolerance for early stopping")


    # parser.add_argument("--tmc-num-samples", default=10, type=int)
    args = parser.parse_args()
    # seed for reproducibility
    set_seed(args.seed)

    # create the output path if it does not exist
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # run analysis
    run_analysis(args)