import os.path

import matplotlib
import matplotlib.pyplot as plt
import torch
import tkinter


def visualize_copy_number_profiles(C: torch.Tensor, save_path=None, pyplot_backend=None,
                                   title_suff: str = '', block=False):
    if save_path is None and pyplot_backend is None:
        matplotlib.use('TkAgg')
    elif save_path is None and pyplot_backend == "default":
        asd = 1 #matplotlib.use(matplotlib.rcParams['backend'])
    elif save_path is None:
        matplotlib.use('TkAgg')

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

    if save_path is None:
        plt.show(block=block)
    else:
        plt.savefig(save_path)


def visualize_mu_tau(mu: torch.Tensor, tau: torch.Tensor, save_path=None, pyplot_backend=None):
    if save_path is None and pyplot_backend is None:
        matplotlib.use('TkAgg')
    elif save_path is None and pyplot_backend == "default":
        matplotlib.use(matplotlib.rcParams['backend'])
    elif save_path is None:
        matplotlib.use('TkAgg')

    N = mu.shape[0]
    fig, axs = plt.subplots(1, 2)
    fig.suptitle('Mu and tau')

    cells = range(1, N+1)
    axs[0].plot(cells, mu)
    axs[0].set_title(f"mu")
    axs[1].plot(cells, tau)
    axs[1].set_title(f"tau")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def visualize_diagnostics(diagnostics_dict: dict, cells_to_vis_idxs=[0], clones_to_vis_idxs=[1], save_path: str = ''):
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
    axs[1, 1].plot(diagnostics_dict["C"][int(max_iter/4), clones_to_vis_idxs[0], :].argmax(dim=-1))
    axs[1, 2].plot(diagnostics_dict["C"][int(max_iter/2), clones_to_vis_idxs[0], :].argmax(dim=-1))
    axs[1, 3].plot(diagnostics_dict["C"][max_iter-1, clones_to_vis_idxs[0], :].argmax(dim=-1))

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


def plot_diagnostics_to_pdf(diagnostics_dict: dict, cells_to_vis_idxs=[0], clones_to_vis_idxs=[1], save_path: str = ''):
    max_iter, N = diagnostics_dict["nu"].shape
    max_iter, K, M, A = diagnostics_dict["C"].shape

    n_rows = 1 + len(clones_to_vis_idxs) + len(cells_to_vis_idxs)
    n_cols = 4
    fig, axs = plt.subplots(n_rows, n_cols)
    fig.suptitle(f"Diagnostics - cells {cells_to_vis_idxs}")
    axs[0, 0].plot(diagnostics_dict["nu"][:, cells_to_vis_idxs[0]])
    axs[0, 0].title(r'$\nu$')
    axs[0, 1].plot(diagnostics_dict["lmbda"][:, cells_to_vis_idxs[0]])
    axs[0, 1].title(r'$\lambda$')
    axs[0, 2].plot(diagnostics_dict["alpha"][:, cells_to_vis_idxs[0]])
    axs[0, 2].title(r'$\alpha$')
    axs[0, 3].plot(diagnostics_dict["beta"][:, cells_to_vis_idxs[0]])
    axs[0, 3].title(r'$\beta$')

    i = 1
    for clone_idx in range(clones_to_vis_idxs):
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

    plt.show()

    fig.savefig(os.path.join(save_path, '.pdf'), bbox_inches='tight')


if __name__ == '__main__':
    K_test, M_test = (5, 10)
    C_test = torch.ones((K_test, M_test))
    C_test[2, int(M_test / 2):] = 2.
    C_test[3, int(M_test / 2):] = 2.
    visualize_copy_number_profiles(C_test)

    N_test = 15
    mu_test = torch.ones(N_test) * 5. + torch.rand(N_test) * 2.0
    tau_test = torch.ones(N_test) * 0.1 * torch.rand(N_test)
    visualize_mu_tau(mu_test, tau_test)
