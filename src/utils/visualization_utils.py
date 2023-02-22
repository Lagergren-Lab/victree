import matplotlib
import matplotlib.pyplot as plt
import torch
import tkinter


def visualize_copy_number_profiles(C: torch.Tensor):
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

    plt.show()


if __name__ == '__main__':
    K_test, M_test = (5, 10)
    C_test = torch.ones((K_test, M_test))
    C_test[2, int(M_test / 2):] = 2.
    C_test[3, int(M_test / 2):] = 2.
    visualize_copy_number_profiles(C_test)
