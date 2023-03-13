"""
Experiments with Directed Slantis
"""
import pandas as pd
import time

import torch

import simul
from utils.config import Config, set_seed
from utils.tree_utils import tree_to_newick
from variational_distributions.var_dists import qT


def kl_divergence(p_dict: dict, q_dict: dict) -> float:
    """
    Computes KL(p||q) = sum_x p(x) * [log(p(x)) - log(q(x))].
    Parameters
    ----------
    p_dict, log-p values for all values that x can take
    q_dict, keys is subset (can be smaller) of p_dict's keys, values are log-probs

    Returns
    -------

    """
    kl = torch.tensor(0.)
    for x, logp in p_dict.items():
        # if q doesn't support x, set very low logq (less than 1 / #values)
        logq = - torch.log(torch.tensor(len(p_dict))) - 1.
        if x in q_dict:
            logq = q_dict[x]
        kl += logp.exp() * (logp - logq)

    return kl.item()


def create_qt(config: Config, uniform: bool = True, n_iter: int = 10, verbose: bool = False):
    # qt with uniformly initialized graph
    qt: qT = qT(config).initialize()
    if verbose:
        print(f"initialized qt")
        print(qt)
    if not uniform:
        # simulate jointq and update qt so to reach skewed distribution
        jointq = simul.generate_dataset_var_tree(config)
        for it in range(n_iter):
            qt.update(jointq.c, jointq.eps)
            if verbose:
                print(f"update iter {it}")
                print(qt)

    return qt


def dslantis_efficiency_experiment(nodes=None,
                                   sample_sizes: list | None = None, uniform_qt: bool = True,
                                   to_file: str = '', verbose: bool = False) -> pd.DataFrame:
    """
Compute KL divergence between exact q(T) and sampled importance weights
with varying number of nodes and sample size.
Output is pandas dataframe or csv file with header: n_nodes, sample_size, kl, time(s)
    """
    if nodes is None:
        nodes = list(range(4, 8))
    if sample_sizes is None:
        sample_sizes = [20, 50, 100, 200, 500, 1000, 2000]

    out_df = pd.DataFrame({'n_nodes': int(), 'sample_size': int(),
                           'kl': float(), 'time(s)': float()}, index=[])
    for n_nodes in nodes:
        config = Config(n_nodes=n_nodes)
        for ss in sample_sizes:
            # re-initialize qt graph weights
            qt = create_qt(config, uniform=uniform_qt, verbose=verbose, n_iter=1)

            if verbose:
                print(f"running n_nodes={n_nodes}, ss={ss}")

            start = time.time()
            dsl_trees, dsl_lw = qt.get_trees_sample(alg='dslantis', sample_size=ss, log_scale=True, torch_tensor=True)
            ttime = time.time() - start
            # save samples with unique values into dict
            dsl_dict = {}
            for t, lw in zip(dsl_trees, dsl_lw):
                t_str = tree_to_newick(t)
                if t_str not in dsl_dict:
                    dsl_dict[t_str] = -torch.tensor(torch.inf)
                dsl_dict[t_str] = torch.logaddexp(dsl_dict[t_str], lw)

            # enumerate exact trees
            trees, logqt = qt.enumerate_trees()
            enum_dict = {tree_to_newick(t): qt for t, qt in zip(trees, logqt)}

            # compute kl divergence
            kl = kl_divergence(p_dict=enum_dict, q_dict=dsl_dict)

            # save values
            curr_df = pd.DataFrame([[kl, n_nodes, ss, ttime]], columns=['kl', 'n_nodes', 'sample_size', 'time(s)'])
            out_df = pd.concat([out_df, curr_df], ignore_index=True)

    print("finish")
    if to_file:
        print(f"file saved to " + to_file)
        out_df.to_csv(to_file)
    return out_df


if __name__ == '__main__':
    set_seed(42)
    # out_df = dslantis_efficiency_experiment(to_file='./sampling_trees_defaults.csv', verbose=True)
    out_df = dslantis_efficiency_experiment(uniform_qt=False, to_file='./sampling_trees_updated_qt.csv')
    print(out_df.head())
