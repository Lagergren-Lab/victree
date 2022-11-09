import logging

import networkx as nx
import torch
from networkx.algorithms.tree import Edmonds


def create_fully_connected_graph(W, root):
    n_nodes = W.shape[0]
    # Create directed graph
    G = nx.DiGraph(directed=True)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            # if W[i, j] == -np.infty:
            #     continue

            if i == root:
                G.add_edge(i, j, weight=W[i, j])

            else:
                G.add_edge(j, i, weight=W[i, j])
                if j != root:
                    G.add_edge(i, j, weight=W[i, j])
    return G


def get_start_arborescence(log_W, alg="edmonds"):
    G = create_fully_connected_graph(log_W)

    if alg == "edmonds":
        edm_alg = Edmonds(G)
        edmonds_tree = edm_alg.find_optimum(kind="max", style="arborescence", preserve_attrs=True)
    else:
        raise NotImplementedError(f"{alg} not implemented yet.")

    return edmonds_tree


def get_ordered_indexes(n_nodes):
    idx = []
    for e1 in range(0, n_nodes):
        for e2 in range(0, n_nodes):
            if e1 != e2: idx.append((e1, e2))

    return idx


def check_cycles(T_copy):
    pass


def sample_arborescence(log_W: torch.Tensor):
    logger = logging.getLogger('sample_arborescence')
    n_nodes = log_W.shape[0]
    T_init: nx.DiGraph = get_start_arborescence(log_W, "edmonds")
    T = T_init
    M = []

    idx_0 = get_ordered_indexes(n_nodes)
    idx_1 = get_ordered_indexes(n_nodes)

    log_T = 0

    for e0 in idx_0:

        # Case e in T
        if e0 in T.edges:
            # Create T_0, T_1 = cut(T, e)
            # Remove the edge from S, create a cut.
            T_copy = T.copy()
            T_copy.remove_edge(*e0)
            T_0, T_1 = nx.connected_components(T_copy)

            # Loop through idx_1 and see if we can connect the components
            for e1 in idx_1:
                # FIXME: build idx1 in order to exclude e0 and include only feasible arcs
                if e1 != e0:
                    # TODO: check nx function
                    G = nx.compose(T_0, T_1)
                    G.add_edge(e1)
                    if nx.is_arborescence(G):
                        # take e1 as alternative
                        T_alternative = G
                        e_alternative = e1
                        break

        else:  # Case: arc is not in T => cycle
            # Add arc to S temporarily
            T_copy = T.copy()
            T_copy.add_edge(*e0, weight=log_W[e0])
            # If the arc creates a cycle, identify the worst arc in the cycle (which is not in M)
            T_alternative, lightest_edge = check_cycles(T_copy)
            # If there is no arc to remove, skip
            # TODO: check 3 new connection cases:
            #   - siblings/cousins: there exists an arc that can be removed
            #   - ancestor -> descendant: there exists an arc that can be removed
            #   - descendant -> ancestor: it's a mess, avoid this (skip)
            #   in any case, the arc to be removed should not be in M
            if lightest_edge is None:
                continue
            else:
                e_alternative = e1

            # Draw from uniform
            U = torch.rand(1)
            log_p_not_accept = T.size(weight="weight") - torch.logsumexp(T.size(weight="weight"),
                                                                         T_alternative.size(weight="weight"))
            accept_alt = U > torch.exp(log_p_not_accept)

            if T.has_edge(*e0):
                # If edge was in the original S, but there were no alternative edges to replace,
                # we chose original S with probability 1. p_keep = 1
                if e_alternative is None:
                    M.append(e0)
                    #self.log_importance_weight += 0
                    #print("\tEdge ", edge, " is in S, we chose original S (no alternatives).\tp_keep: ", 1)

                # If edge was in the original S and we chose the alternative S
                elif accept_alt:
                    T = T_alternative
                    #print("\tEdge ", edge, " is in S, we chose alternative S.\tp_keep: ", p_keep)

                # If edge was in the original T and we chose the original T
                else:
                    M.append(e0)
                    log_T += log_p_not_accept
                    #print("\tEdge ", edge, " is in S, we chose original S.\tp_keep: ", p_keep)
            else:
                # If edge was in the alternative S and we chose the alternative S
                if not accept_alt:
                    T = T_alternative
                    #print("\tEdge ", edge, " is not in S, we chose alternative S.\tp_keep: ", p_keep)
                # If edge was in the alternative S and we chose the original S. Nothing changes.
                else:
                    #print("\tEdge ", edge, " is not in S, we chose original S.\tp_keep: ", p_keep)
                    pass

    return T, log_T
