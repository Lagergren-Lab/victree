import argparse
import os

import torch

from analysis import qT_analysis, qC_analysis
from utils import factory_utils, data_handling, visualization_utils
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
    if args.data_path is not None:
        data_handler = DataHandler(args.data_path)
        obs = data_handler.norm_reads

    # ---
    # Create configuration object and load checkpoint
    # ---
    checkpoint_data = data_handling.read_checkpoint(args.checkpoint_path)
    config = factory_utils.construct_config_from_checkpoint_data(checkpoint_data)


    # ---
    # Run analysis
    # ---
    if args.victree:
        raise NotImplementedError
    if args.qT:
        q_T = factory_utils.construct_qT_from_model_output_data(checkpoint_data, config)
        qT_analysis.edge_probability_analysis(q_T, args.tree_sample_size)
    if args.qZ:
        raise NotImplementedError
    if args.qC:
        victree = factory_utils.construct_victree_object_from_model_output_and_data(checkpoint_data, obs, config)
        victree_fixed_tree = qC_analysis.train_on_fixed_tree(victree=victree, n_iter=50)
        qC_marginals_argmax = torch.argmax(victree_fixed_tree.q.c.single_filtering_probs, dim=-1)
        visualization_utils.visualize_copy_number_profiles(qC_marginals_argmax)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
        description="VIC-Tree analysis")
    parser.add_argument("-i", "--input_data", dest="data_path",
                        type=validate_path, default='../../data/x_data/signals_SPECTRUM-OV-014.h5',
                        help="output model file", metavar="FILE")
    parser.add_argument("-m", "--input_checkpoint", dest="checkpoint_path",
                        type=validate_path, default='../../output/checkpoint_k6a7n1105m6206.h5',
                        help="output model file", metavar="FILE")
    parser.add_argument("-o", "--output", dest="out_dir",
                        help="output dir", metavar="DIR", default="./output")
    parser.add_argument("--victree", action="store_true", help="Run full victree analysis")
    parser.add_argument("--qT", action="store_true", help="Run qT analysis")
    parser.add_argument("--qZ", action="store_true", help="Run qZ analysis")
    parser.add_argument("--qC", action="store_true", help="Run qC analysis")
    parser.add_argument("-s", "--seed", default=42, type=int, help="RNG seed")
    parser.add_argument("-K", "--n-nodes", default=5, type=int, help="number of nodes/clones")
    parser.add_argument("-A", "--n-states", default=7, type=int, help="number of characters/copy number states")
    parser.add_argument("-S", "--step-size", default=.1, type=float, help="step-size for partial updates")
    parser.add_argument("-L", "--tree-sample-size", default=10, type=int, help="number of sampled arborescences")


    # parser.add_argument("--tmc-num-samples", default=10, type=int)
    args = parser.parse_args()
    # seed for reproducibility
    set_seed(args.seed)

    # create the output path if it does not exist
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # run analysis
    run_analysis(args)