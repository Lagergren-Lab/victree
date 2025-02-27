import torch
import math
import numpy as np
import scipy.special as sp_spec
import scipy.stats as sp_stats

from utils import visualization_utils


def remap_tensor(x: torch.Tensor, perm) -> torch.float:
    index = torch.LongTensor(perm)
    y = torch.zeros_like(x)
    y[index] = x
    return y


def write_inference_output_no_ground_truth(victree, y, test_dir_path, file_name_prefix='', tree=None):
    config = victree.config
    N = config.n_cells
    M = config.chain_length
    K = config.n_nodes
    A = config.n_states
    visualization_utils.visualize_qC_qZ_and_obs(victree.q.c, victree.q.z, y,
                                                save_path=test_dir_path +
                                                          f'/{file_name_prefix}qC_qZ_plot.png')
    visualization_utils.visualize_subclonal_structures_qC_qZ_and_obs(victree.q.c, victree.q.z, y,
                                                            save_path=test_dir_path +
                                                          f'/{file_name_prefix}no_clonal_qC_qZ_plot.png')

    visualization_utils.visualize_qMuTau(victree.q.mt, save_path=test_dir_path + f'/{file_name_prefix}qMuTau_plot.png')
    if tree is not None:
        visualization_utils.draw_graph(tree, save_path=test_dir_path + '/true_tree_plot.png')
    return None
