import os.path
import pathlib
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as f

import simul
from variational_distributions.joint_dists import FixedTreeJointDist
from utils import tree_utils, visualization_utils
from utils.config import Config
from variational_distributions.var_dists import qC, qZ, qPi, qMuTau, qEpsilonMulti


def get_two_node_tree():
    T_1 = nx.DiGraph()
    T_1.add_edge(0, 1)
    return T_1


def get_tree_three_nodes_balanced():
    T_1 = nx.DiGraph()
    T_1.add_edge(0, 1)
    T_1.add_edge(0, 2)
    return T_1


def get_tree_three_nodes_chain():
    T_2 = nx.DiGraph()
    T_2.add_edge(0, 1)
    T_2.add_edge(1, 2)
    return T_2


def get_tree_K_nodes_random(K) -> nx.DiGraph:
    T = nx.DiGraph()
    T.add_edge(0, 1)
    nodes_in_T = [0, 1]
    nodes_not_in_T = list(range(2, K))
    for i in range(len(nodes_not_in_T)):
        parent = np.random.choice(nodes_in_T)
        child = np.random.choice(nodes_not_in_T)
        T.add_edge(parent, child)
        nodes_in_T.append(child)
        nodes_not_in_T.remove(child)

    return T


def get_tree_K_nodes_one_level(K):
    T = nx.DiGraph()
    for k in range(K):
        T.add_edge(0, k)

    return T


def get_random_q_C(M, A):
    q_C_init = torch.rand(A)
    q_C_transitions_unnormalized = torch.rand((M - 1, A, A))
    q_C_transitions = f.normalize(q_C_transitions_unnormalized, p=1, dim=2)
    return q_C_init, q_C_transitions


def get_root_q_C(M, A):
    q_C_init = torch.zeros(A)
    q_C_init[2] = 1
    q_C_transitions = torch.zeros((M - 1, A, A))
    diag_A = torch.ones(A)
    for m in range(M - 1):
        q_C_transitions[m] = torch.diag(diag_A, 0)
    return q_C_init, q_C_transitions


def simul_data_pyro_full_model(data, n_cells, n_sites, n_copy_states, tree: nx.DiGraph,
                               mu_0=torch.tensor(1.),
                               lambda_0=torch.tensor(1.),
                               alpha0=torch.tensor(1.),
                               beta0=torch.tensor(1.),
                               a0=torch.tensor(1.0),
                               b0=torch.tensor(20.0),
                               dir_alpha0=torch.tensor(1.0)
                               ):
    # FIXME: change to simul.simulate_full_dataset
    model_tree_markov_full = simul.model_tree_markov_full
    unconditioned_model = poutine.uncondition(model_tree_markov_full)
    C, y, z, pi, mu, tau, eps = unconditioned_model(data, n_cells, n_sites, n_copy_states, tree,
                                                    mu_0,
                                                    lambda_0,
                                                    alpha0,
                                                    beta0,
                                                    a0,
                                                    b0,
                                                    dir_alpha0)

    return C, y, z, pi, mu, tau, eps


def simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree: nx.DiGraph,
                                  nu_0=1.,
                                  lambda_0=1.,
                                  alpha0=1.,
                                  beta0=1.,
                                  a0=1.0,
                                  b0=20.0,
                                  dir_alpha0=1.0,
                                  simulate_raw_reads=True,
                                  ):
    n_nodes = len(tree.nodes)
    config = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites)
    output_sim = simul.simulate_full_dataset(config, eps_a=a0, eps_b=b0, mu0=nu_0, lambda0=lambda_0, alpha0=alpha0,
                                             beta0=beta0, dir_alpha=dir_alpha0, tree=tree, raw_reads=simulate_raw_reads)
    y = output_sim['obs']
    C = output_sim['c']
    z = output_sim['z']
    pi = output_sim['pi']
    mu = output_sim['mu']
    tau = output_sim['tau']
    eps = output_sim['eps']
    eps0 = output_sim['eps0']

    return y, C, z, pi, mu, tau, eps, eps0


def simulate_quadruplet_data(M, A, tree: nx.DiGraph, eps_a, eps_b, eps_0):
    output_sim = simul.simulate_quadruplet_data(M, A, tree, eps_a, eps_b, eps_0)
    y = output_sim['obs']
    c = output_sim['c']
    mu = output_sim['mu']
    tau = output_sim['tau']
    eps = output_sim['eps']
    eps0 = output_sim['eps0']

    return y, c, mu, tau, eps, eps0


def get_quadtruplet_tree() -> nx.DiGraph:
    quad_topology = nx.DiGraph()
    quad_topology.add_edge(0, 1)
    quad_topology.add_edge(1, 2)
    quad_topology.add_edge(1, 3)
    return quad_topology


def generate_test_dataset_fixed_tree() -> FixedTreeJointDist:
    # obs with 15 cells, 5 each to different clone
    # in order, clone 0, 1, 2
    cells_per_clone = 10
    mm = 1  # change this to increase length
    chain_length = mm * 10  # total chain length shouldn't be more than 100, ow eps too small
    cfg = Config(n_nodes=3, n_states=5, n_cells=3 * cells_per_clone, chain_length=chain_length, wis_sample_size=2,
                 step_size=0.8, debug=True)
    # obs with 15 cells, 5 each to different clone
    # in order, clone 0, 1, 2
    true_cn_profile = torch.tensor(
        [[2] * 10 * mm,
         [2] * 4 * mm + [3] * 6 * mm,
         [1] * 3 * mm + [3] * 2 * mm + [2] * 3 * mm + [3] * 2 * mm]
        # [3] * 10]
    )
    # cell assignments
    true_z = torch.tensor([0] * cells_per_clone +
                          [1] * cells_per_clone +
                          [2] * cells_per_clone)
    true_pi = torch.nn.functional.one_hot(true_z, num_classes=cfg.n_nodes).float()

    cell_cn_profile = true_cn_profile[true_z, :]

    # mean and precision
    nu, lmbda = torch.tensor([1, 10])  # randomize mu for each cell with these hyperparameters
    true_mu = torch.randn(cfg.n_cells) / torch.sqrt(lmbda) + nu
    obs = (cell_cn_profile * true_mu[:, None]).T.clamp(min=0)

    true_eps = {
        (0, 1): 1. / (cfg.chain_length - 1),
        (0, 2): 3. / (cfg.chain_length - 1)
    }

    # give true values to the other required dists
    fix_qc = qC(cfg, true_params={
        "c": true_cn_profile
    })

    fix_qz = qZ(cfg, true_params={
        "z": true_z
    })

    fix_qeps = qEpsilonMulti(cfg, true_params={
        "eps": true_eps
    })

    fix_qmt = qMuTau(cfg, true_params={
        "mu": true_mu,
        "tau": torch.ones(cfg.n_cells) * lmbda
    })

    fix_qpi = qPi(cfg, true_params={
        "pi": torch.ones(cfg.n_nodes) / 3.
    })

    fix_tree = nx.DiGraph()
    fix_tree.add_edges_from([(0, 1), (0, 2)], weight=.5)

    joint_q = FixedTreeJointDist(cfg, fix_qc, fix_qz, fix_qeps, fix_qmt, fix_qpi, fix_tree, obs)
    return joint_q


def save_test_data(seed, tree, C, y, z, pi, mu, tau, eps):
    K, M = C.shape
    A = int(torch.max(C)) + 1
    N = mu.shape[0]
    parent_folder = "data/" if "tests" in os.getcwd() else "tests/data/"
    dir_name = f"K{K}_N{N}_M{M}_A{A}_Seed{seed}"
    path = parent_folder + dir_name if "data" not in os.getcwd() else dir_name
    if os.path.exists(path):
        raise FileExistsError

    os.mkdir(path)
    pickle.dump(tree, open(path + '/tree.pickle', 'wb'))
    torch.save(C, path + '/C.pt')
    torch.save(y, path + '/y.pt')
    torch.save(z, path + '/z.pt')
    torch.save(pi, path + '/pi.pt')
    torch.save(mu, path + '/mu.pt')
    torch.save(tau, path + '/tau.pt')
    torch.save(eps, path + '/eps.pt')

    visualization_utils.visualize_copy_number_profiles(C, save_path=path + "/CN_profiles.png")
    visualization_utils.visualize_mu_tau(mu, tau, save_path=path + "/mu_tau_plot.png")


def load_test_data(seed, K, M, N, A):
    parent_folder = "data/" if "tests" in os.getcwd() else "tests/data/"
    dir_name = f"K{K}_N{N}_M{M}_A{A}_Seed{seed}/"
    path = parent_folder + dir_name
    if not os.path.exists(path):
        raise FileNotFoundError

    tree = pickle.load(path + 'tree.pickle')
    C = torch.load(path + 'C.pt')
    y = torch.load(path + 'y.pt')
    z = torch.load(path + 'z.pt')
    pi = torch.load(path + 'pi.pt')
    mu = torch.load(path + 'mu.pt')
    tau = torch.load(path + 'tau.pt')
    eps = torch.load(path + 'eps.pt')
    return tree, C, y, z, pi, mu, tau, eps


def create_test_output_catalog(config=None, test_specific_string=None, base_dir="./test_output"):
    path = base_dir + "/" + test_specific_string
    path += "" if config is None else f"/K{config.n_nodes}_N{config.n_cells}_M{config.chain_length}_A{config.n_states}"
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        print("Dir already exists. Overwriting contents.")
    return path

def create_experiment_output_catalog(experiment_path, base_dir="./test_output"):
    path = base_dir + "/" + experiment_path
    try:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        print("Dir already exists. Risk of overwriting contents.")
    return path


def get_two_sliced_marginals_from_one_slice_marginals(marginals, A, offset=None):
    K, M = marginals.shape
    two_sliced_marginals = torch.zeros((K, M-1, A, A))
    marginals_one_hot = torch.nn.functional.one_hot(marginals, A)

    for u in range(K):
        for m in range(0, M-1):
            a_1 = marginals[u, m]
            a_2 = marginals[u, m+1]
            two_sliced_marginals[u, m, a_1, a_2] = 1.

    if offset is not None:
        two_sliced_marginals += offset
        two_sliced_marginals = two_sliced_marginals / two_sliced_marginals.sum(dim=(2, 3), keepdims=True)

    return two_sliced_marginals