import networkx as nx
import torch


class GenerativeModel():

    # TODO: set the right parameters/defaults
    # i.e. set obs loc and obs scale
    def __init__(self, tree: nx.DiGraph = nx.DiGraph(), epsilon: torch.Tensor = torch.tensor(0.1),
            cell_to_clone_assignment = torch.empty(1), copy_number_model = None,
                 obs_loc = 100, obs_scale = 10):
        self.tree = tree
        self.epsilon = epsilon
        self.cell_to_clone_assignment = cell_to_clone_assignment
        self.copy_number_model = copy_number_model
        self.obs_model = torch.distributions.Normal(loc = obs_loc, scale = obs_scale)

    def evaluate(self):
        return 0

