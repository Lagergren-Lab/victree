import networkx as nx
import torch


class GenerativeModel():

    def __init__(self, tree: nx.DiGraph, epsilon: torch.Tensor, cell_to_clone_assignment, copy_number_model):
        self.tree = tree
        self.epsilon = epsilon
        self.cell_to_clone_assignment = cell_to_clone_assignment
        self.copy_number_model = copy_number_model



    def evaluate(self):
        return 0