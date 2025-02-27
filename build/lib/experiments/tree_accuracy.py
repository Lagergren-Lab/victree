from utils.data_handling import read_vi_gt
from utils.evaluation import best_mapping
from sampling_trees.sampling_trees import trees_percentiles_from_experiment


# FIXME: not working, gt tree does not even appear in samples
if __name__=='__main__':
    checkpoint_file = "/Users/zemp/phd/scilife/cpt_experiments/tree_exp/checkpoint_k5a7n500m1000.h5"
    simul_file = "/Users/zemp/phd/scilife/cpt_experiments/tree_exp/simul_k5a7n500m1000e40-10000d50mt1-10-500-50.h5"
    vi, gt = read_vi_gt(checkpoint_file, simul_file)
    print(trees_percentiles_from_experiment(vi, gt))
