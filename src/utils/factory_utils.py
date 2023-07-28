import os.path

import torch

from inference.victree import VICTree
from utils.config import Config
from variational_distributions import var_dists
import data_handling
from variational_distributions.joint_dists import VarTreeJointDist


def construct_config_from_checkpoint_data(checkpoint_data):
    qZ = checkpoint_data['qZ']['pi']
    A, N, n_iter = qZ.shape
    qC = checkpoint_data['qC'] if 'qC' in checkpoint_data.keys() else checkpoint_data['qCMultiChrom']
    n_iter, K, M, A = qC['single_filtering_probs'].shape
    return Config(n_cells=N, chain_length=M, n_nodes=K, n_states=A)


def construct_q_from_checkpoint_data(checkpoint_data, obs, config=None):
    config = construct_config_from_checkpoint_data(checkpoint_data) if config is None else config
    qZ_params = checkpoint_data['qZ']
    qZ = var_dists.qZ(config)
    qZ.initialize()
    qZ.pi = qZ_params
    qC_params = checkpoint_data['qC']
    qC = var_dists.qC(config)
    qC.initialize()
    qC.single_filtering_probs = qC_params
    qpi_params = checkpoint_data['qPi']
    qpi = var_dists.qPi(config)
    qpi.initialize()
    qpi.concentration_param = qpi_params
    qT = construct_qT_from_checkpoint_data(checkpoint_data, config)
    qeps = checkpoint_data['qEpsilonMulti']
    qmutau = checkpoint_data['qMuTau']
    return VarTreeJointDist(config, obs, qC, qZ, qpi, qT, qeps, qmutau)


def construct_qT_from_checkpoint_data(checkpoint_data, config=None):
    config = construct_config_from_checkpoint_data(checkpoint_data) if config is None else config
    qT_params = checkpoint_data['qT']
    qT_weight_matrix = torch.tensor(qT_params['weight_matrix'])
    qT = var_dists.qT(config)
    qT.initialize()
    qT._weight_matrix = qT_weight_matrix[-1, :, :]
    return qT


def construct_victree_object_from_checkpoint_file(file_path, data_path):
    checkpoint_data = data_handling.read_checkpoint(file_path)

    config = construct_config_from_checkpoint_data(checkpoint_data)
    q = construct_q_from_checkpoint_data(checkpoint_data)
    obs = data_handling.load_h5_pseudoanndata(data_path)
    victree = VICTree(config, q, obs)
    return victree


if __name__ == '__main__':
    file_path = os.path.join("./../../output", "checkpoint_k6a7n1105m6206.h5")
    data_path = os.path.join("./../../data/x_data", "signals_SPECTRUM-OV-014.h5")

    checkpoint_data = data_handling.read_checkpoint(file_path)
    qT = construct_qT_from_checkpoint_data(checkpoint_data)
    print(qT.weight_matrix)
    # victree = construct_victree_object_from_checkpoint_file(file_path, data_path)
