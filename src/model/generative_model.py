import networkx as nx
import torch


class GenerativeModel():

    # TODO: set the right parameters/defaults
    def __init__(self, tree: nx.DiGraph = nx.DiGraph(), epsilon: torch.Tensor = torch.tensor(0.1),
            cell_to_clone_assignment = torch.empty(1), copy_number_model = None):
        self.tree = tree
        self.epsilon = epsilon
        self.cell_to_clone_assignment = cell_to_clone_assignment
        self.copy_number_model = copy_number_model



    def evaluate(self):
        return 0
