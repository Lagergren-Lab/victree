import os.path

from inference.victree import VICTree
from utils.config import Config
from variational_distributions import var_dists
import data_handling
from variational_distributions.joint_dists import VarTreeJointDist


def construct_config_from_checkpoint_data(checkpoint_data):
    qZ = checkpoint_data['qZ']['pi']
    A, N, K = qZ.shape
    qC = checkpoint_data['qC'] if 'qC' in checkpoint_data.keys() else checkpoint_data['qCMultiChrom']
    M, K, A = qC['single_filter_probs'].shape
    return Config(n_cells=N, chain_length=M, n_nodes=K, n_states=A)


def construct_q_from_checkpoint_data(checkpoint_data, config=None):
    config = construct_config_from_checkpoint_data(checkpoint_data) if config is None else config
    qZ_params = checkpoint_data['qZ']
    qZ = var_dists.qZ()
    qZ.initialize()
    qC_params = checkpoint_data['qC']
    qpi = checkpoint_data['qpi']
    qT = checkpoint_data['qT']
    qeps = checkpoint_data['qEpsilonMulti']
    qmutau = checkpoint_data['qMuTau']
    return VarTreeJointDist(qC, qZ, qpi, qT, qeps, qmutau)


def construct_victree_object_from_checkpoint_file(file_path, data_path):

    checkpoint_data = data_handling.read_checkpoint(file_path)

    config = construct_config_from_checkpoint_data(checkpoint_data)
    q = construct_q_from_checkpoint_data(checkpoint_data)
    obs = data_handling.load_h5_pseudoanndata(data_path)
    victree = VICTree(config, q, obs)
    return victree



if __name__ == '__main__':
    file_path = os.path.join("./../../tests/test_output", "checkpoint_k3a3n30m5.h5")
    data_path = os.path.join("./../../datasets", "simul_k5a7n300m1000e1-50d10mt1-10-500-50.h5")

    victree = construct_victree_object_from_checkpoint_file(file_path, data_path)