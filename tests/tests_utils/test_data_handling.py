import unittest

import anndata
import numpy as np
from networkx.lazy_imports import os

import simul
from inference.victree import VICTree
from utils.config import Config
from utils.data_handling import DataHandler
from tests.data.generate_data import generate_2chr_adata
from variational_distributions.joint_dists import VarTreeJointDist
from variational_distributions.var_dists import qCMultiChrom
from utils.data_handling import write_output


class dataHandlingTestCase(unittest.TestCase):

    def setUp(self) -> None:

        self.output_dir = "./test_output"
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def test_datahandler_impute_nans(self):
        chain_length_with_nans = 200
        # method remove nans
        cfg = Config(chain_length=chain_length_with_nans, n_cells=100)
        chr_df = simul.generate_chromosome_binning(cfg.chain_length, method='uniform', n_chr=5)
        data = simul.simulate_full_dataset(cfg, chr_df=chr_df,
                                           nans=True, cne_length_factor=10)
        dh = DataHandler(adata=data['adata'], impute_nans='remove', config=cfg)
        # check that some bins have been correctly removed
        self.assertLess(dh.get_anndata().n_vars, chain_length_with_nans)
        # check that config is consistent with new number of bins
        self.assertEqual(dh.get_anndata().n_vars, cfg.chain_length)

    def test_write_output(self):
        out_file = os.path.join(self.output_dir, 'out_test.h5')
        # run victree
        config = Config(n_nodes=4, n_cells=20, n_states=4,
                        n_run_iter=3, sieving_size=2, n_sieving_iter=2)
        adata = generate_2chr_adata(config)
        data_handler = DataHandler(adata=adata)
        obs = data_handler.norm_reads
        qc = qCMultiChrom(config)
        q = VarTreeJointDist(config, obs, qc=qc).initialize()
        victree = VICTree(config, q, obs, data_handler)

        victree.run()

        write_output(victree, out_file, anndata=True)

        # read anndata and assert fields
        out_adata = anndata.read_h5ad(out_file)

        # check fields
        for l in ['victree-cn-viterbi', 'victree-cn-marginal']:
            self.assertTrue(l in out_adata.layers)
        for l in ['victree-mu', 'victree-tau', 'victree-clone', 'victree-loglik']:
            self.assertTrue(l in out_adata.obs)
        for l in ['victree-clone-probs']:
            self.assertTrue(l in out_adata.obsm)
        for l in ['victree-cn-sprobs']:
            self.assertTrue(l in out_adata.varm)
        for l in ['victree-tree-graph']:
            self.assertTrue(l in out_adata.uns)

        # assert sizes
        self.assertEqual(out_adata.layers['victree-cn-viterbi'].shape, (config.n_cells, config.chain_length))

        # assert values
        self.assertTrue(np.all(out_adata.obs['victree-loglik'] < 0))





            


        

        
