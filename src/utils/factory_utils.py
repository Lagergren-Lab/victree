import os.path

import torch

from inference.victree import VICTree
from utils.config import Config
from variational_distributions import var_dists
#import data_handling
from variational_distributions.joint_dists import VarTreeJointDist


def construct_config_from_checkpoint_data(checkpoint_data):
    qZ = checkpoint_data['qZ']['pi']
    N, K = qZ.shape
    if 'qC' in checkpoint_data.keys():
        qC = checkpoint_data['qC']
        K, M, A = qC['eta2'].shape
        M += 1  # eta2 of shape M-1
    else:
        qC = checkpoint_data['qCMultiChrom']
        M = 0
        A = qC['0']['eta2'].shape[2]
        chr_idx = []
        idxs = [int(key) for key in qC.keys()]
        idxs.sort()
        for idx in idxs:
            M += qC[str(idx)]['eta2'].shape[1] + 1
            if idx != idxs[-1]:
                chr_idx.append(M)

        return Config(n_cells=N, chain_length=M, n_nodes=K, n_states=A, chromosome_indexes=chr_idx)

    return Config(n_cells=N, chain_length=M, n_nodes=K, n_states=A)


def construct_q_from_checkpoint_data(model_output_h5_file, obs, config=None):
    config = construct_config_from_checkpoint_data(model_output_h5_file) if config is None else config
    qZ = construct_qZ_from_model_output_data(model_output_h5_file, config)
    if 'qC' in model_output_h5_file.keys():
        qC = construct_qC_from_model_output_data(model_output_h5_file, config)
    else:
        qC = construct_qCMultiChrome_from_model_output_data(model_output_h5_file, config)
    qpi = construct_qPi_from_model_output_data(model_output_h5_file, config)
    qT = construct_qT_from_model_output_data(model_output_h5_file, config)
    qeps = construct_qEpsilonMulti_from_model_output_data(model_output_h5_file, config)
    qmutau = construct_qMuTau_from_model_output_data(model_output_h5_file, config)
    return VarTreeJointDist(config, obs, qC, qZ, qT, qeps, qmutau, qpi)


def construct_qT_from_checkpoint_data(checkpoint_data, config=None):
    config = construct_config_from_checkpoint_data(checkpoint_data) if config is None else config
    qT_params = checkpoint_data['qT']
    qT_weight_matrix = torch.tensor(qT_params['weight_matrix'][-1])
    qT = var_dists.qT(config)
    qT.initialize()
    qT._weight_matrix = qT_weight_matrix
    return qT


def construct_qT_from_model_output_data(model_output_data, config=None):
    config = construct_config_from_checkpoint_data(model_output_data) if config is None else config
    qT_params = model_output_data['qT']
    qT_weight_matrix = torch.tensor(qT_params['weight_matrix'])
    qT = var_dists.qT(config)
    qT.initialize()
    qT._weight_matrix = qT_weight_matrix
    return qT


def construct_qC_from_model_output_data(model_output_data, config=None):
    config = construct_config_from_checkpoint_data(model_output_data) if config is None else config
    qC_params = model_output_data['qC']
    eta1 = torch.tensor(qC_params['eta1'])
    eta2 = torch.tensor(qC_params['eta2'])
    qC = var_dists.qC(config)
    qC.initialize()
    qC.eta1 = eta1
    qC.eta2 = eta2
    return qC


def construct_qCMultiChrome_from_model_output_data(model_output_data, config=None):
    config = construct_config_from_checkpoint_data(model_output_data) if config is None else config
    qC_params = model_output_data['qCMultiChrom']
    qCMultiChrom = var_dists.qCMultiChrom(config)
    qCMultiChrom.initialize()
    for key in qC_params.keys():
        eta1 = torch.tensor(qC_params[key]['eta1'])
        eta2 = torch.tensor(qC_params[key]['eta2'])
        qCMultiChrom.qC_list[int(key)].eta1 = eta1
        qCMultiChrom.qC_list[int(key)].eta2 = eta2
    return qCMultiChrom


def construct_qZ_from_model_output_data(model_output_data, config=None):
    config = construct_config_from_checkpoint_data(model_output_data) if config is None else config
    qZ_params = torch.tensor(model_output_data['qZ']['pi'])
    qz = var_dists.qZ(config)
    qz.initialize()
    qz.pi = qZ_params
    return qz


def construct_qPi_from_model_output_data(model_output_data, config=None):
    config = construct_config_from_checkpoint_data(model_output_data) if config is None else config
    qpi_params = torch.tensor(model_output_data['qPi']['concentration_param'])
    qpi = var_dists.qPi(config)
    qpi.initialize()
    qpi.concentration_param = qpi_params
    return qpi


def construct_qMuTau_from_model_output_data(model_output_data, config=None):
    config = construct_config_from_checkpoint_data(model_output_data) if config is None else config
    qMuTau_params = model_output_data['qMuTau']
    qmt = var_dists.qMuTau(config)
    qmt.initialize()
    qmt.nu = torch.tensor(qMuTau_params['nu'])
    qmt.lmbda = torch.tensor(qMuTau_params['lmbda'])
    qmt.alpha = torch.tensor(qMuTau_params['alpha'])
    qmt.beta = torch.tensor(qMuTau_params['beta'])
    return qmt


def construct_qEpsilonMulti_from_model_output_data(model_output_data, config=None):
    config = construct_config_from_checkpoint_data(model_output_data) if config is None else config
    qEps_params = model_output_data['qEpsilonMulti']
    qeps = var_dists.qEpsilonMulti(config)
    qeps.initialize()
    loaded_alphas = torch.tensor(qEps_params['alpha'])
    loaded_betas = torch.tensor(qEps_params['beta'])
    loaded_alphas_dict = {e: loaded_alphas[e] for e in qeps.alpha_dict.keys()}
    loaded_betas_dict = {e: loaded_betas[e] for e in qeps.alpha_dict.keys()}
    qeps.alpha_dict = loaded_alphas_dict
    qeps.beta_dict = loaded_betas_dict
    return qeps


def construct_victree_object_from_model_output_and_data(model_output_h5_file, obs, config=None):
    config = config if config is None else construct_config_from_checkpoint_data(model_output_h5_file)
    q = construct_q_from_checkpoint_data(model_output_h5_file, obs, config)
    victree = VICTree(config, q, obs)
    victree.compute_elbo()
    return victree


if __name__ == '__main__':
    file_path = os.path.join("./../../output", "checkpoint_k6a7n1105m6206.h5")
    data_path = os.path.join("./../../data/x_data", "signals_SPECTRUM-OV-014.h5")

    checkpoint_data = data_handling.read_checkpoint(file_path)
    qT = construct_qT_from_checkpoint_data(checkpoint_data)
    print(qT.weight_matrix)
    # victree = construct_victree_object_from_checkpoint_file(file_path, data_path)
