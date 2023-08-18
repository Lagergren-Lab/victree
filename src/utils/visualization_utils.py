import os.path
import pickle
import sys
import argparse

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import torch
from networkx.drawing.nx_agraph import graphviz_layout


def visualize_copy_number_profiles(C: torch.Tensor, save_path=None, pyplot_backend=None,
                                   title_suff: str = '', block=False):
    if save_path is None and pyplot_backend is None:
        matplotlib.use('module://backend_interagg')
    elif save_path is None and pyplot_backend == "default":
        matplotlib.use(matplotlib.rcParams['backend'])
    elif save_path is None:
        matplotlib.use(pyplot_backend)
    else:
        matplotlib.use('module://backend_interagg')

    if len(C.shape) > 2 and C.shape[2] > 1:
        C = torch.argmax(C, dim=2)  # convert one-hot-encoding to categories

    K, M = C.shape
    A_max = int(torch.max(C))
    n_col = 2
    n_rows = int(K / n_col) + 1 if K % n_col != 0 else int(K / n_col)
    fig, axs = plt.subplots(n_rows, n_col)
    title = "CN profile " + title_suff
    fig.suptitle(title)

    sites = range(1, M + 1)
    for k in range(K):
        C_k = C[k, :]
        if k % n_col == 0:
            col_count = 0
        else:
            col_count += 1

        axs[int(k / n_col), col_count].plot(sites, C_k)
        axs[int(k / n_col), col_count].set_title(f'k = {k}')

    plt.show(block=block)

    user_continue = input("Check plot and determine if C looks plausible. Continue? (y/n)")
    if user_continue == 'y':
        if save_path is not None:
            fig.savefig(save_path + 'c_simulated.png')
    else:
        raise Exception('Simulated C rejected by user.')


def visualize_copy_number_profiles_ipynb(C, title_suff: str = ''):
    plt.interactive(True)
    if len(C.shape) > 2 and C.shape[2] > 1:
        C = torch.argmax(C, dim=2)  # convert one-hot-encoding to categories

    K, M = C.shape
    A_max = int(np.max(C))
    n_col = 2
    n_rows = int(K / n_col) + 1 if K % n_col != 0 else int(K / n_col)
    fig, axs = plt.subplots(n_rows, n_col)
    title = "CN profile " + title_suff
    fig.suptitle(title)

    sites = range(1, M + 1)
    for k in range(K):
        C_k = C[k, :]
        if k % n_col == 0:
            col_count = 0
        else:
            col_count += 1

        axs[int(k / n_col), col_count].plot(sites, C_k)
        axs[int(k / n_col), col_count].set_title(f'k = {k}')

    plt.show()


def visualize_copy_number_profiles_of_multiple_sources(multi_source_SxKxM_array: np.array, save_path=None,
                                                       pyplot_backend=None,
                                                       title_suff: str = '', block=False):
    """
    Plots S lines in K subplots with x-axis of length M.
    Used e.g. to plot simulated C, qC marginals and qC marginals after fixed tree tuning for all clones.
    """
    if save_path is None and pyplot_backend is None:
        matplotlib.use('module://backend_interagg')
    elif save_path is None and pyplot_backend == "default":
        matplotlib.use(matplotlib.rcParams['backend'])
    elif save_path is None:
        matplotlib.use(pyplot_backend)

    n_sources, K, M = multi_source_SxKxM_array.shape
    A_max = int(np.max(multi_source_SxKxM_array))
    n_col = 2
    n_rows = int(K / n_col) + 1 if K % n_col != 0 else int(K / n_col)
    fig, axs = plt.subplots(n_rows, n_col)
    title = "CN profile " + title_suff
    fig.suptitle(title)
    labels = [str(i) for i in range(n_sources)]
    sites = range(1, M + 1)
    for k in range(K):
        if k % n_col == 0:
            col_count = 0
        else:
            col_count += 1
        for source in range(n_sources):
            C_k = multi_source_SxKxM_array[source, k, :]
            axs[int(k / n_col), col_count].plot(sites, C_k, label=f'{source}')
            axs[int(k / n_col), col_count].set_title(f'k = {k}')

        if k == 0:
            axs[int(k / n_col), col_count].legend(labels)

    if save_path is None:
        plt.show(block=block)
    else:
        plt.savefig(save_path)


def visualize_qC_true_C_qZ_and_true_Z(c, qc, z, qz, save_path=None,
                                      title_suff: str = '', pyplot_backend=None):
    """
    Plots S lines in K subplots with x-axis of length M.
    Used e.g. to plot simulated C, qC marginals and qC marginals after fixed tree tuning for all clones.
    """
    if save_path is None and pyplot_backend is None:
        matplotlib.use('module://backend_interagg')  # Used for plotting in PyCharm scientific mode
    elif save_path is None and pyplot_backend == "default":
        matplotlib.use(matplotlib.rcParams['backend'])
    elif save_path is None:
        matplotlib.use(pyplot_backend)

    K, M, A = qc.single_filtering_probs.shape
    qc_viterbi = qc.single_filtering_probs.argmax(dim=-1)
    qz_mean_assignment = qz.pi.mean(dim=0)
    z_mean = torch.nn.functional.one_hot(z, num_classes=K).float().mean(dim=0)
    n_col = 2
    n_rows = int(K / n_col) + 1 if K % n_col != 0 else int(K / n_col)
    fig, axs = plt.subplots(n_rows, n_col)
    title = "CN profile " + title_suff
    fig.suptitle(title)
    labels = ['true c', 'qc viterbi']
    sites = range(1, M + 1)

    # Print format for z and qz
    print_z_mean = ["{:.2f}".format(e) for e in z_mean.tolist()]
    print_qz_mean = ["{:.2f}".format(e) for e in qz_mean_assignment.tolist()]

    # Plot c and qc viterbi path in K subplots
    for k in range(K):
        if k % n_col == 0:
            col_count = 0
        else:
            col_count += 1

        row = int(k / n_col)
        axs[row, col_count].plot(sites, c[k])
        axs[row, col_count].plot(sites, qc_viterbi[k])
        axs[row, col_count].set_title(f'k = {k}')
        axs[row, col_count].text(0.1, 0.8, print_z_mean[k], fontsize=7, transform=axs[row, col_count].transAxes,
                                 bbox={'facecolor': 'blue', 'alpha': 0.4, 'pad': 2})
        axs[row, col_count].text(0.1, 0.6, print_qz_mean[k], fontsize=7, transform=axs[row, col_count].transAxes,
                                 bbox={'facecolor': 'orange', 'alpha': 0.4, 'pad': 2})

        if k == 0:
            axs[int(k / n_col), col_count].legend(labels)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def visualize_qC_qZ_and_obs(qc, qz, obs, save_path=None,
                                      title_suff: str = '', pyplot_backend=None):
    """
    Plots S lines in K subplots with x-axis of length M.
    Used e.g. to plot simulated C, qC marginals and qC marginals after fixed tree tuning for all clones.
    """
    if save_path is None and pyplot_backend is None:
        matplotlib.use('module://backend_interagg')  # Used for plotting in PyCharm scientific mode
    elif save_path is None and pyplot_backend == "default":
        matplotlib.use(matplotlib.rcParams['backend'])
    elif save_path is None:
        matplotlib.use(pyplot_backend)

    K, M, A = qc.single_filtering_probs.shape
    qc_viterbi = qc.single_filtering_probs.argmax(dim=-1)
    qz_mean_assignment = qz.pi.mean(dim=0)
    n_col = 2
    n_rows = int(K / n_col) + 1 if K % n_col != 0 else int(K / n_col)
    fig, axs = plt.subplots(n_rows, n_col)
    title = "CN profile " + title_suff
    fig.suptitle(title)
    labels = ['qc viterbi, obs']
    sites = range(1, M + 1)

    # Print format for z and qz
    print_qz_mean = ["{:.2f}".format(e) for e in qz_mean_assignment.tolist()]

    # Plot c and qc viterbi path in K subplots
    for k in range(K):
        if k % n_col == 0:
            col_count = 0
        else:
            col_count += 1

        row = int(k / n_col)
        #axs[row, col_count].plot(sites, obs[:, qz.pi.argmax(dim=-1) == k])
        axs[row, col_count].plot(sites, qc_viterbi[k])
        axs[row, col_count].set_title(f'k = {k}')
        axs[row, col_count].text(0.1, 0.6, print_qz_mean[k], fontsize=7, transform=axs[row, col_count].transAxes,
                                 bbox={'facecolor': 'orange', 'alpha': 0.4, 'pad': 2})

        if k == 0:
            axs[int(k / n_col), col_count].legend(labels)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def visualize_observations_copy_number_profiles_of_multiple_sources(multi_source_SxKxM_array, obs, assignments,
                                                                    save_path=None, pyplot_backend=None,
                                                                    title_suff: str = ''):
    """
    Plots S lines in K subplots with x-axis of length M.
    Used e.g. to plot simulated C, qC marginals and qC marginals after fixed tree tuning for all clones.
    """
    if save_path is None and pyplot_backend is None:
        matplotlib.use('module://backend_interagg')
    elif save_path is None and pyplot_backend == "default":
        matplotlib.use(matplotlib.rcParams['backend'])
    elif save_path is None:
        matplotlib.use(pyplot_backend)

    n_sources, K, M = multi_source_SxKxM_array.shape
    A_max = int(np.max(multi_source_SxKxM_array)) if type(multi_source_SxKxM_array) is np.ndarray \
        else int(torch.max(multi_source_SxKxM_array))
    n_col = 2
    n_rows = int(K / n_col) + 1 if K % n_col != 0 else int(K / n_col)
    fig, axs = plt.subplots(n_rows, n_col)
    title = "CN profile " + title_suff
    fig.suptitle(title)
    labels = ['1', '2', '3']
    sites = range(1, M + 1)
    for k in range(K):
        if k % n_col == 0:
            col_count = 0
        else:
            col_count += 1
        for source in range(n_sources):
            C_k = multi_source_SxKxM_array[source, k, :]
            axs[int(k / n_col), col_count].plot(sites, C_k, label=f'{source}')
            axs[int(k / n_col), col_count].set_title(f'k = {k}')

        axs[int(k / n_col), col_count].plot(obs[:, assignments == k], 'o', alpha=0.1)
        if k == 0:
            axs[int(k / n_col), col_count].legend(labels)

    if save_path is None:
        plt.show(block=False)
    else:
        plt.savefig(save_path)


def visualize_mu_tau(mu: torch.Tensor, tau: torch.Tensor, save_path=None, pyplot_backend=None):
    if save_path is None and pyplot_backend is None:
        matplotlib.use('module://backend_interagg')
    elif save_path is None and pyplot_backend == "default":
        matplotlib.use(matplotlib.rcParams['backend'])
    elif save_path is None:
        matplotlib.use('module://backend_interagg')

    N = mu.shape[0]
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Mu and tau')

    cells = range(1, N + 1)
    axs[0].plot(cells, mu)
    axs[0].set_title(f"mu")
    axs[1].plot(cells, tau)
    axs[1].set_title(f"tau")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def visualize_diagnostics(diagnostics_dict: dict, cells_to_vis_idxs=[0], clones_to_vis_idxs=[1], save_path: str = ''):
    # FIXME: change from diagnostics dict to params_history values in each var dist
    plt.switch_backend("TkAgg")
    max_iter, N = diagnostics_dict["nu"].shape
    max_iter, K, M, A = diagnostics_dict["C"].shape
    n_rows = 4
    n_cols = 4
    fig, axs = plt.subplots(n_rows, n_cols)
    fig.suptitle(f"Diagnostics - cells {cells_to_vis_idxs}")
    axs[0, 0].plot(diagnostics_dict["nu"][:, cells_to_vis_idxs[0]])
    axs[0, 1].plot(diagnostics_dict["lmbda"][:, cells_to_vis_idxs[0]])
    axs[0, 2].plot(diagnostics_dict["alpha"][:, cells_to_vis_idxs[0]])
    # print all even if not updated (constant line)
    # axs[0, 2].plot(torch.arange(0, max_iter), diagnostics_dict["alpha"][cells_to_vis_idxs[0]].expand(max_iter))
    axs[0, 3].plot(diagnostics_dict["beta"][:, cells_to_vis_idxs[0]])

    axs[1, 0].plot(diagnostics_dict["C"][0, clones_to_vis_idxs[0], :].argmax(dim=-1))
    axs[1, 1].plot(diagnostics_dict["C"][int(max_iter / 4), clones_to_vis_idxs[0], :].argmax(dim=-1))
    axs[1, 2].plot(diagnostics_dict["C"][int(max_iter / 2), clones_to_vis_idxs[0], :].argmax(dim=-1))
    axs[1, 3].plot(diagnostics_dict["C"][max_iter - 1, clones_to_vis_idxs[0], :].argmax(dim=-1))

    for i, cell_idx in enumerate(cells_to_vis_idxs):
        if i >= n_cols:
            break
        axs[2, i].plot(diagnostics_dict["Z"][:, cell_idx])

    axs[3, 0].plot(diagnostics_dict["pi"])
    axs[3, 1].plot(diagnostics_dict["eps_a"][:, 0, 1])
    axs[3, 2].plot(diagnostics_dict["eps_b"][:, 0, 1])

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_diagnostics_to_pdf(diagnostics_dict: dict,
                            cells_to_vis_idxs=[0],
                            clones_to_vis_idxs=[1],
                            edges_to_vis_idxs=[(0, 1)],
                            save_path: str = ''):
    max_iter, N = diagnostics_dict["nu"].shape
    max_iter, K, M, A = diagnostics_dict["C"].shape

    # number of rows: 3 from qMuTau, pi plot and extra C_m^u plot
    n_rows = 3 + len(clones_to_vis_idxs) + int(len(cells_to_vis_idxs) / 4 + 1) + int(len(edges_to_vis_idxs) / 2 + 1)
    n_cols = 4
    rows_per_page = 6
    with PdfPages(save_path) as pdf:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        fig.tight_layout(pad=2.0)  # space between subplots
        fig.suptitle(f"Diagnostics - cells {cells_to_vis_idxs} - clones {clones_to_vis_idxs}")
        axs[0, 0].plot(diagnostics_dict["nu"][:, cells_to_vis_idxs[0]])
        axs[0, 0].set_title(r'$\nu$')
        axs[0, 1].plot(diagnostics_dict["lmbda"][:, cells_to_vis_idxs[0]])
        axs[0, 1].set_title(r'$\lambda$')
        axs[0, 2].plot(diagnostics_dict["alpha"][:, cells_to_vis_idxs[0]])
        axs[0, 2].set_title(r'$\alpha$')
        axs[0, 3].plot(diagnostics_dict["beta"][:, cells_to_vis_idxs[0]])
        axs[0, 3].set_title(r'$\beta$')

        i = 1
        for clone_idx in clones_to_vis_idxs:
            axs[i, 0].plot(diagnostics_dict["C"][0, clone_idx, :].argmax(dim=-1))
            axs[i, 1].plot(diagnostics_dict["C"][int(max_iter / 4), clone_idx, :].argmax(dim=-1))
            axs[i, 2].plot(diagnostics_dict["C"][int(max_iter / 2), clone_idx, :].argmax(dim=-1))
            axs[i, 3].plot(diagnostics_dict["C"][max_iter - 1, clone_idx, :].argmax(dim=-1))

            axs[i, 0].set_title(f'iter 0 - clone {clone_idx}')
            axs[i, 1].set_title(f'iter {int(max_iter / 4)} - clone {clone_idx}')
            axs[i, 2].set_title(f'iter {int(max_iter / 2)} - clone {clone_idx}')
            axs[i, 3].set_title(f'iter {max_iter - 1} - clone {clone_idx}')
            i += 1

        # C_m^u for single u and m over all iter
        m0 = 0
        m1 = int(M * 1 / 3)
        m2 = int(M * 2 / 3)
        m3 = M - 1
        clone_idx = clones_to_vis_idxs[0]
        axs[i, 0].plot(diagnostics_dict["C"][:, clone_idx, m0].argmax(dim=-1))
        axs[i, 0].set_title(f'Clone {clone_idx} over iterations and m= {m0}')
        axs[i, 1].plot(diagnostics_dict["C"][:, clone_idx, m1].argmax(dim=-1))
        axs[i, 1].set_title(f'Clone {clone_idx}, m= {m1}')
        axs[i, 2].plot(diagnostics_dict["C"][:, clone_idx, m2].argmax(dim=-1))
        axs[i, 2].set_title(f'Clone {clone_idx}, m= {m2}')
        axs[i, 3].plot(diagnostics_dict["C"][:, clone_idx, m3].argmax(dim=-1))
        axs[i, 3].set_title(f'Clone {clone_idx}, m= {m3}')

        i += 1
        j = 0
        for cell_idx in cells_to_vis_idxs:
            if j >= n_cols:
                j = 0
                i += 1
            for k in range(K):
                axs[i, j].plot(diagnostics_dict["Z"][:, cell_idx, k], label=f'{k}')
                axs[i, j].set_title(f'Z for cell {cell_idx}')

            axs[i, j].legend()

            j += 1

        # visualize pi
        i += 1
        axs[i, 0].plot(diagnostics_dict["pi"])
        axs[i, 0].set_title('pi over iterations')
        axs[i, 0].legend()

        i += 1
        # visualize epsilon
        even = True
        for a in edges_to_vis_idxs:
            if even:
                axs[i, 0].plot(diagnostics_dict["eps_a"][:, a[0], a[1]])
                axs[i, 1].plot(diagnostics_dict["eps_b"][:, a[0], a[1]])
                axs[i, 0].set_title(f"Epsilon a param for arc: {a[0]}, {a[1]}")
                axs[i, 1].set_title(f"Epsilon b param for arc: {a[0]}, {a[1]}")
                even = False
            else:
                axs[i, 2].plot(diagnostics_dict["eps_a"][:, a[0], a[1]])
                axs[i, 3].plot(diagnostics_dict["eps_b"][:, a[0], a[1]])
                axs[i, 2].set_title(f"Epsilon a param for arc: {a[0]}, {a[1]}")
                axs[i, 3].set_title(f"Epsilon b param for arc: {a[0]}, {a[1]}")
                even = True
                i += 1

        plt.close(fig)
        pdf.savefig(fig)

        # plot Pi dist
        fig, ax = plt.subplots()
        ax.stackplot(range(max_iter), *[diagnostics_dict["pi"][:, i] for i in range(K)], labels=range(K))
        ax.legend(loc='upper left')
        plt.title("qPi")
        plt.close(fig)
        pdf.savefig(fig)


def visualize_T_given_true_tree_and_distances(w_T_list, distances, g_T_list=None, save_path=None):
    sorted_dist = np.sort(distances)
    sorted_dist_idx = np.argsort(distances)
    x_axis = np.arange(len(distances))
    w_T_list_sorted = [w_T_list[i] for i in sorted_dist_idx]
    fig = plt.plot(x_axis, w_T_list_sorted, 'o')
    distances_break_points_indexes = np.where(sorted_dist[1:] - sorted_dist[:-1] > 0)[0] + 2
    labels = ['' if i not in distances_break_points_indexes else sorted_dist[i] for i in x_axis]
    labels[0] = sorted_dist[0]
    plt.xlabel("Distance to true tree")
    plt.ylabel("w(T)")
    plt.xticks(ticks=x_axis, labels=labels)
    if save_path is not None:
        plt.savefig(save_path + '/w_T_and_T_to_true_tree_distance_plot.png')
    if g_T_list is not None:
        plt.close()
        g_T_list_sorted = [g_T_list[i] for i in sorted_dist_idx]
        fig = plt.plot(x_axis, g_T_list_sorted, 'o')
        plt.xlabel("Distance to true tree")
        plt.ylabel("g(T)")
        plt.xticks(ticks=x_axis, labels=labels)
        plt.savefig(save_path + '/g_T_and_T_to_true_tree_distance_plot.png')

    return fig


def visualize_mu_tau_true_and_q(mu, tau, qmt, save_path=None):
    N = mu.shape[0]
    x_axis = np.arange(0, N)
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('qMuTau vs true mu and tau')
    axs[0, 0].plot(x_axis, mu)
    axs[0, 0].plot(x_axis, qmt.nu)
    axs[0, 0].set_title('mu')
    axs[0, 0].legend(['true mu', 'E_q[mu]'])
    axs[0, 1].plot(x_axis, tau)
    axs[0, 1].plot(x_axis, qmt.exp_tau())
    axs[0, 1].set_title('tau')
    axs[0, 1].legend(['true tau', 'E_q[tau]'])

    # Errors
    axs[1, 0].plot(x_axis, torch.abs(mu - qmt.nu))
    axs[1, 0].set_title('mu error')
    axs[1, 1].plot(x_axis, torch.abs(tau - qmt.exp_tau()))
    axs[1, 1].set_title('tau error')

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def draw_graph(G: nx.DiGraph, save_path=None):
    pos = graphviz_layout(G, prog="dot")

    f: plt.figure.Figure = plt.figure(figsize=(5, 5))
    nx.draw(G, pos=pos, with_labels=True, ax=f.add_subplot(111))
    if save_path is not None:
        f.savefig(save_path, format="png")
    else:
        plt.show()


def visualize_and_save_T_plots(save_path, true_tree, T_list, w_T_list, distances, g_T_list_unique=None):
    with PdfPages(save_path + '/qT_visualization.pdf') as pdf:
        fig = visualize_T_given_true_tree_and_distances(w_T_list, distances, g_T_list_unique, save_path=save_path)


def test_visualization():
    K_test, M_test, A_test = (5, 10, 7)
    C_test = torch.ones((K_test, M_test))
    C_test[2, int(M_test / 2):] = 2.
    C_test[3, int(M_test / 2):] = 2.
    # visualize_copy_number_profiles(C_test)

    N_test = 15
    mu_test = torch.ones(N_test) * 5. + torch.rand(N_test) * 2.0
    tau_test = torch.ones(N_test) * 0.1 * torch.rand(N_test)
    # visualize_mu_tau(mu_test, tau_test)

    n_iter = 10
    diag_C = torch.ones((n_iter, K_test, M_test, A_test))
    diag_Z = torch.ones((n_iter, N_test, K_test))
    diag_nu = torch.ones((n_iter, N_test))
    diag_lmbda = torch.ones((n_iter, N_test))
    diag_alpha = torch.ones((n_iter, N_test))
    diag_beta = torch.ones((n_iter, N_test))

    diag_epsilon_a = torch.ones((n_iter, K_test, K_test))
    diag_epsilon_b = torch.ones((n_iter, K_test, K_test))
    diag_pi = torch.ones((n_iter, K_test))

    diag_dict = {"C": diag_C, "Z": diag_Z, "nu": diag_nu, "lmbda": diag_lmbda, "alpha": diag_alpha, "beta": diag_beta,
                 "pi": diag_pi, "eps_a": diag_epsilon_a, "eps_b": diag_epsilon_b}
    save_path = '../../tests/test_output/test.pdf'
    plot_diagnostics_to_pdf(diagnostics_dict=diag_dict,
                            cells_to_vis_idxs=[0, 3, 5, 10, 12, 14],
                            clones_to_vis_idxs=[0, 1, 3],
                            edges_to_vis_idxs=[(0, 1), (1, 2), (2, 4)], save_path=save_path)


def edge_type(e):
    try:
        u, v = map(int, e.split(','))
        return u, v
    except Exception:
        raise argparse.ArgumentTypeError("Edges must be u,v")


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--pickle-file',
                     type=str,
                     required=True, help="pickle file path (binary pickle)")
    cli.add_argument('--out-path',
                     type=str,
                     required=True, help="output path e.g. ./diagnostics.pdf")
    cli.add_argument('--cells-list',
                     type=int,
                     nargs='*',
                     default=[0, 1, 2], help="list of cells e.g. 0 2 3")
    cli.add_argument('--clones-list',
                     type=int,
                     nargs='*',
                     default=[0, 1, 2], help="list of clones e.g. 0 2 3")
    cli.add_argument('--edges-list',
                     type=edge_type,
                     nargs='*',
                     default=[(0, 1), (0, 2), (1, 2), (2, 1)],
                     help="list of edges with comma separated clone names e.g. 0,1 1,2 2,1")
    # if empty list, print all related to clones
    args = cli.parse_args()

    with open(args.pickle_file, 'rb') as f:
        diag_dict = pickle.load(f)
        plot_diagnostics_to_pdf(diagnostics_dict=diag_dict,
                                cells_to_vis_idxs=args.cells_list,
                                clones_to_vis_idxs=args.clones_list,
                                edges_to_vis_idxs=args.edges_list,
                                save_path=args.out_path)
