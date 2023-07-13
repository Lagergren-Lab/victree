import networkx as nx
import torch

from model.cell_to_clone_model import CellToCloneModel
from model.copy_tree_model import CopyTreeModel
from model.edge_independent_comutation_rate_model import EdgeIndependentComutationProbModel
from model.observational_models.cell_specific_baseline_gc_normal_observational_model import \
    CellSpecificBaselineGCNormalObsModel
from utils.config import Config


class MultiChromosomeGenerativeModel():

    def __init__(self, config: Config):
        self.config = config
        N, M, K, A = (config.n_cells, config.chain_length, config.n_nodes, config.n_states)
        self.chromes_idx = config.chromosome_indexes
        self.n_chromes = config.n_chromosomes
        self.copy_number_evolution_models = []   # one model per chromosome
        for i in range(self.n_chromes):
            if i == self.n_chromes-1:
                M_chr_i = M - self.chromes_idx[i-1]
            elif i == 0:
                M_chr_i = self.chromes_idx[i]
            else:
                M_chr_i = self.chromes_idx[i] - self.chromes_idx[i-1]

            self.copy_number_evolution_models.append(CopyTreeModel(M_chr_i, K, A))  # p(C | T, eps)

        self.cell_to_clone_assignment_model = CellToCloneModel(N, K)  # p(Z, pi)
        self.emission_model = CellSpecificBaselineGCNormalObsModel(N, M)  # p(Y, Psi | Z, C)
        self.co_mutation_model = EdgeIndependentComutationProbModel(K)  # p(eps | T)

    def simulate_data(self, T: nx.DiGraph, a0, b0, eps0_a, eps0_b, delta, nu0, lambda0, alpha0, beta0):
        # simulate chromosome copy numbers independently
        eps, eps0 = self.co_mutation_model.simulate_data(T, a0, b0, eps0_a, eps0_b)
        c_all = []
        for chrome_idx in range(self.config.n_chromosomes):
            c_temp = self.copy_number_evolution_models[chrome_idx].simulate_data(T, eps0, eps)
            c_all.append(c_temp)

        c = torch.cat(c_all, dim=1)

        z, pi = self.cell_to_clone_assignment_model.simulate_data(delta)
        y, mu, tau = self.emission_model.simulate_data(c, z, nu0, lambda0, alpha0, beta0)
        out_simul = {
            'obs': y,
            'c': c,
            'z': z,
            'pi': pi,
            'mu': mu,
            'tau': tau,
            'eps': eps,
            'eps0': eps0,
        }
        return out_simul







