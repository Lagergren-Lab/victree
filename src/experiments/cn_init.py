import torch
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

import simul
from tests.utils_testing import get_tree_K_nodes_one_level
from utils.config import Config
from variational_distributions.var_dists import qMuTau, qCMultiChrom, qZ, qEpsilonMulti


def run(M: int, K: int, A: int, N: int, n_iter: int, n_datasets: int):
    # set params
    a0 = 5.
    b0 = 300.

    # simulate datasets
    print("simulating dataset...")
    config = Config(n_cells=N, n_nodes=K, n_states=A, chain_length=M, n_run_iter=n_iter, debug=True)
    # for dsi in range(n_datasets):
    fix_joint_q = simul.generate_dataset_var_tree(config, chrom='real')

    # set up qmt, cell-qc
    cell_config = Config(n_cells=N, n_nodes=N+1, n_states=A, n_run_iter=n_iter, step_size=0.2,
                         chain_length=config.chain_length, chromosome_indexes=config.chromosome_indexes, debug=True)
    qmt = qMuTau(cell_config, nu_prior=1., lambda_prior=10., alpha_prior=50., beta_prior=5.)
    qmt.initialize(method='prior')

    cell_qc = qCMultiChrom(cell_config)
    cell_qc.initialize(method='random')

    qeps = qEpsilonMulti(cell_config, alpha_prior=a0, beta_prior=b0)
    qeps.initialize(method='prior')

    # run inference with qmt,cell-qc
    fixed_z = torch.arange(1, N + 1)
    qz_cells_fix = qZ(cell_config, true_params={'z': fixed_z})
    tree = get_tree_K_nodes_one_level(N + 1)

    print("running inference...")
    for i in range(n_iter):
        qmt.update(cell_qc, qz_cells_fix, fix_joint_q.obs)
        cell_qc.update(fix_joint_q.obs, qeps, qz_cells_fix, qmt,
                       [tree], [1.])
        qeps.update([tree], torch.tensor([1.]), cell_qc)
        if (i+1) % 10 == 0:
            print(f"it {i}, elbo {cell_qc.compute_elbo([tree], [1.], q_eps=qeps)}")

    print("running k-means")
    # cluster copy numbers with kmeans (on viterbi)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(cell_qc.get_viterbi()[1:])
    c_labels = kmeans.labels_

    print(f"ARI for KMeans after cell-cn estimation: {adjusted_rand_score(c_labels, fix_joint_q.z.true_params['z'].numpy())}")

    # cluster copy numbers with kmeans (on viterbi)
    obs_kmeans = KMeans(n_clusters=K, random_state=0).fit(fix_joint_q.obs.T.numpy())
    obs_labels = obs_kmeans.labels_

    print(f"ARI for KMeans with obs: {adjusted_rand_score(obs_labels, fix_joint_q.z.true_params['z'].numpy())}")

    # initialize new qc with average
    # build eta1 and eta2 from cell_qc
    qc = qCMultiChrom(config)

    for i, chr_qc in enumerate(qc.qC_list):
        # init for each chromosome the average tensor
        avg_param = {'eta1': torch.empty_like(chr_qc.eta1), 'eta2': torch.empty_like(chr_qc.eta2)}
        for k in range(K):
            cells_k = c_labels == k
            print(f"{cells_k.sum()} cells in clone {k}")
            # average of eta parameters over cells assigned to the same cluster
            avg_param['eta1'][k, ...] = cell_qc.qC_list[i].eta1[cells_k].mean(dim=0)
            avg_param['eta2'][k, ...] = cell_qc.qC_list[i].eta2[cells_k].mean(dim=0)

        chr_qc.initialize(method='fixed', **avg_param)

    # continue inference with qz also

    # save plots/data
    print("run finished")


if __name__=='__main__':
    params = {
        'M': 500,
        'K': 5,
        'A': 7,
        'N': 100,
        'n_iter': 20,
        'n_datasets': 2
    }
    run(**params)
