import matplotlib
import matplotlib.pyplot as plt
import torch
import tkinter


def visualize_copy_number_profiles(C: torch.Tensor, save_path=None, pyplot_backend=None):
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
    fig.suptitle('Copy numbers profiles')

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
        plt.show()
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
