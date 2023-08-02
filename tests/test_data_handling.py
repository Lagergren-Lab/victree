import unittest

from matplotlib.pyplot import logging
from networkx.lazy_imports import os

import simul
from inference.victree import VICTree
from utils.config import Config
from utils.data_handling import read_sc_data, DataHandler
from tests.data.generate_data import generate_2chr_adata
from variational_distributions.joint_dists import VarTreeJointDist
from variational_distributions.var_dists import qCMultiChrom
from utils.data_handling import write_output


class dataHandlingTestCase(unittest.TestCase):

    def setUp(self) -> None:

        self.output_dir = "./test_output"
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def test_datahandler(self):
        pass

    def test_write_output(self):
        out_file = os.path.join(self.output_dir, 'out_test.h5ad')
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
        out_dh = DataHandler(out_file)
        out_adata = out_dh.get_anndata()

        # check fields
        for l in ['victree-cn-viterbi', 'victree-cn-marginal']:
            self.assertTrue(l in out_adata.layers)
        for l in ['victree-mu', 'victree-tau', 'victree-clone']:
            self.assertTrue(l in out_adata.obs)
        for l in ['victree-clone-probs']:
            self.assertTrue(l in out_adata.obsm)
        for l in ['victree-cn-sprobs']:
            self.assertTrue(l in out_adata.varm)
        for l in ['victree-tree-graph']:
            self.assertTrue(l in out_adata.uns)

        # assert sizes
        self.assertEqual(out_adata.layers['victree-cn-viterbi'].shape, (config.n_cells, config.chain_length))





            


        

        
