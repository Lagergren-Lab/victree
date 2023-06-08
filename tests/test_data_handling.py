import unittest

from matplotlib.pyplot import logging
from networkx.lazy_imports import os
from utils.data_handling import read_sc_data

class dataHandlingTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.correct_data_file = """c1 c2 c3 c5
        ENSG01 1 1 1 1
        ENSG02 20 124 155 3333
        ENSG03 3 0 0 0"""

        # with missing read at line 2 ENSG02
        self.wrong_df = """c1 c2 c3 c5
        ENSG01 1 1 1 1
        ENSG02 20 124 3333
        ENSG03 3 0 0 0"""

    def test_read_file(self):
        # write both datasets to files
        cfname = "tmp_correct_data.txt"
        wfname = "tmp_wrong_data.txt"
        try:
            with open(cfname, "w+") as cf:
                cf.write(self.correct_data_file)

            with open(wfname, "w+") as wf:
                wf.write(self.wrong_df)

            cell_names, gene_ids, obs = read_sc_data(cfname)
            self.assertEqual(len(cell_names), obs.shape[1])
            self.assertEqual(len(gene_ids), obs.shape[0])
            with self.assertRaises(RuntimeError) as re_info:
                logging.error(re_info)
                _ = read_sc_data(wfname)
        except Exception:
            raise
        finally:
            # make sure the temporary files are removed
            os.remove(cfname)
            os.remove(wfname)
            


        

        
