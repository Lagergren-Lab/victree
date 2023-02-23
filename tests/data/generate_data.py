import torch

import tests
from tests import utils_testing
from tests.utils_testing import simul_data_pyro_full_model
from utils import visualization_utils


def generate_data_balanced_tree(seed, n_cells, n_sites, n_copy_states):
        torch.manual_seed(seed)
        tree = tests.utils_testing.get_tree_three_nodes_balanced()
        n_nodes = len(tree.nodes)
        data = torch.ones((n_sites, n_cells))
        dir_alpha = torch.tensor([3., 5., 5.])
        C, y, z, pi, mu, tau, eps = simul_data_pyro_full_model(data,
                                                               n_cells, n_sites, n_copy_states,
                                                               tree,
                                                               mu_0=torch.tensor(10.),
                                                               lambda_0=torch.tensor(1.),
                                                               alpha0=torch.tensor(10.),
                                                               beta0=torch.tensor(40.),
                                                               a0=torch.tensor(1.0),
                                                               b0=torch.tensor(10.0),
                                                               dir_alpha0=dir_alpha)
        print(f"Simulated data")
        vis_clone_idx = z[80]
        print(f"C: {C[vis_clone_idx, 40]} y: {y[80, 40]} z: {z[80]} \n"
              f"pi: {pi} mu: {mu[80]} tau: {tau[80]} eps: {eps}")
        #visualization_utils.visualize_copy_number_profiles(C, pyplot_backend='defualt')
        #visualization_utils.visualize_mu_tau(mu, tau, pyplot_backend='defualt')

        #if input("Save? (y/n)") == "y":
        try:
            utils_testing.save_test_data(seed, tree, C, y, z, pi, mu, tau, eps)
        except FileExistsError:
            print(f"File exists: K{n_nodes}_N{n_cells}_M{n_sites}_A{n_copy_states}_Seed{seed}")

if __name__ == '__main__':
    seeds = [2, 3, 5, 6, 7, 8, 9]
    n_cells = 1000
    n_sites = 300
    n_copy_states = 7
    for seed in seeds:
        generate_data_balanced_tree(seed, n_cells, n_sites, n_copy_states)
