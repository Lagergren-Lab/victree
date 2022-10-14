import pytest
import numpy as np
import logging

from variational_distributions.variational_distribution import VariationalDistribution
from model.generative_model import GenerativeModel
from inference.copy_tree import CopyTree

def test_elbo_decrease_exception():
    
    p = GenerativeModel()
    q = VariationalDistribution()
    copy_tree = CopyTree(p, q, q)

    copy_tree.elbo = -100. 
    # compute_elbo currently outputs -1000
    # elbo is not updated until step() is not fully implemented

    with pytest.raises(ValueError) as ve_info:
        logging.info(str(ve_info))
        copy_tree.run(10)

