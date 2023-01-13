from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as f

from inference.copy_tree import VarDistFixedTree
from model.generative_model import GenerativeModel


def fixed_T_comparisons(p: GenerativeModel, q: VarDistFixedTree):
    pass
