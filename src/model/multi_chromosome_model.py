import networkx as nx

from utils.config import Config


class MultiChromosomeGenerativeModel():

    def __init__(self, config: Config):
        self.config = config
        self.HMM = CopyTree()
        self.MHMM = CellSpecificBaselineGCNormalObsModel()
        self.co_mutation_model = EpsilonModel()
        self.tree_topology_model = UniformTreeModel()

    def simulate_data(self, T: nx.DiGraph):
        # simulate chromosome copy numbers independently
        eps0, eps = self.co_mutation_model.simulate_data()
        for chrome_idx in range(self.config.n_chromosomes):
            c = self.HMM.simulate_data(T, eps0, eps)







