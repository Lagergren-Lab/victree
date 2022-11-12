import copy
import logging

import networkx as nx
import torch
from matplotlib import pyplot as plt
from networkx.algorithms.tree import Edmonds
from networkx.drawing.nx_pydot import graphviz_layout


def create_fully_connected_graph(W, root):
    n_nodes = W.shape[0]
    # Create directed graph
    G = nx.DiGraph(directed=True)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):

            if i == root:
                G.add_edge(i, j, weight=W[i, j])  # networkx Edmonds determines root node based on in degree = 0

            else:
                G.add_edge(i, j, weight=W[i, j])
                if j != root:
                    G.add_edge(j, i, weight=W[j, i])
    return G


def select_internal_root(log_W_root):
    """
    Selects a root node for the arborescence connected to the healthy root clone.
    :param: log_W_root
    :return root: root node
    """
    Cat = torch.distributions.Categorical(probs=torch.exp(log_W_root))
    root = Cat.sample() + 1  # 0 index corresponds to healthy root
    return int(root)


def get_start_arborescence(log_W, log_W_root, alg="edmonds"):

    if alg == "edmonds":
        root = select_internal_root(log_W_root)
        arborescence = get_edmonds_arborescence(log_W, root)
    else:
        raise NotImplementedError(f"{alg} not implemented yet.")

    return arborescence


def get_edmonds_arborescence(log_W: torch.Tensor, root: int):
    G = create_fully_connected_graph(torch.exp(log_W), root=root)
    edm_alg = Edmonds(G)
    edmonds_tree = edm_alg.find_optimum(kind="max", style="arborescence", preserve_attrs=True)
    return edmonds_tree


def get_ordered_indexes(n_nodes):
    idx = []
    for e1 in range(0, n_nodes):
        for e2 in range(1, n_nodes):  # exclude node to root case
            if e1 != e2: idx.append((e1, e2))

    return idx


def check_cycles(T_copy):
    pass


def draw_graph(G: nx.DiGraph):
    pos = graphviz_layout(G, prog="dot")
    nx.draw(G, pos=pos, with_labels=True)
    plt.show()


def is_root(param):
    pass


def sample_arborescence(log_W: torch.Tensor, root: int, debug=False):
    logger = logging.getLogger('sample_arborescence')
    n_nodes = log_W.shape[0]
    # T_init: nx.DiGraph = get_start_arborescence(log_W, log_W_root, alg="edmonds")
    # T = T_init
    S = []
    S_nodes = set(())
    roots = set(())
    children = set(())
    log_S = 0
    S_arborescence = nx.DiGraph()
    log_W_copy = copy.deepcopy(log_W)
    including_weight = torch.max(log_W) + torch.log(torch.tensor(n_nodes))

    idx_0 = get_ordered_indexes(n_nodes)

    log_T = 0
    n_tries = 0
    while len(S) < n_nodes - 1 or n_tries > 100:
        n_tries += 1
        for e in idx_0:
            log_W_0 = copy.deepcopy(log_W_copy)  # guarantee S included
            #log_W_0[:, e[1]] = -torch.inf  # guarantee no co-parents <--- redundant?
            log_W_0[e] = including_weight  # guarantee e included
            T_0 = get_edmonds_arborescence(log_W_0, root)
            T_0.edges[e]['weight'] = log_W[e]  # set W(e) to actual weight
            if debug:
                assert set(S) <= set(T_0.edges)
                assert e in set(T_0.edges)
            for s in S:
                T_0.edges[s]['weight'] = log_W[s]  # set W(s) to actual weight

            log_W_1 = copy.deepcopy(log_W_copy)  # guarantee S included
            log_W_1[e] = -torch.inf  # guarantee e excluded
            T_1: nx.DiGraph = get_edmonds_arborescence(log_W_1, root)
            if debug:
                assert set(S) <= set(T_1.edges)
                assert e not in set(T_1.edges)
            for s in S:
                T_1.edges[s]['weight'] = log_W[s]  # set W(s) to actual weight

            log_T0 = T_0.size(weight="weight")
            log_T1 = T_1.size(weight="weight")
            log_sum_T0_T1 = torch.logaddexp(log_T0, log_T1)
            theta = torch.exp(log_T0 - log_sum_T0_T1)
            U = torch.rand(1)
            if theta > U:
                # choose e
                S.append(e)
                u, v = e
                children.add(v)
                if u not in children:
                    roots.add(u)
                roots.discard(v)
                """
                if u not in S_nodes:
                    roots.append(u)
                if v in roots:
                    roots.remove(v)
                """

                S_nodes.add(u)
                S_nodes.add(v)
                S_arborescence.add_edge(u, v)

                log_S += torch.log(theta)
                log_W_copy[e] = including_weight  # guarantee e included
                cycles_inducing_arcs = [(v, s) for s in children if s != v]  # filter out possible cycles

                # filter out component root connection
                components = nx.weakly_connected_components(S_arborescence)
                sub_arbs = []
                for comp in components:
                    sub_arbs.append(comp)
                    if v in comp:
                        self_root = roots.intersection(comp).pop()
                        for child in children.intersection(comp):
                            cycles_inducing_arcs.append((child, self_root))

                for a_cycle in cycles_inducing_arcs:
                    if a_cycle in idx_0:    # replace if statements with faster system
                        idx_0.remove(a_cycle)
                    if (a_cycle[1], a_cycle[0]) in idx_0:
                        idx_0.remove((a_cycle[1], a_cycle[0]))
                idx_0[:] = [x for x in idx_0 if x[1] != v]  # filter out possible co-parent arcs

            else:
                continue

    return S_arborescence, log_S

def sample_arborescence_root(log_W: torch.Tensor, log_W_root: torch.Tensor):
    logger = logging.getLogger('sample_arborescence')
    n_nodes = log_W.shape[0]
    # T_init: nx.DiGraph = get_start_arborescence(log_W, log_W_root, alg="edmonds")
    # T = T_init
    S = []
    log_S = copy.deepcopy(log_W)

    idx_0 = get_ordered_indexes(n_nodes)
    # idx_1 = get_ordered_indexes(n_nodes)

    log_T = 0
    n_tries = 0
    while len(S) < n_nodes - 1 or n_tries > 100:
        n_tries += 1
        for e in idx_0:
            log_W_0 = copy.deepcopy(log_S)  # guarantee S included
            log_W_0[e] = torch.inf  # guarantee e included
            T_0 = get_start_arborescence(log_W_0, log_W_root, alg="edmonds")
            T_0.edges[e]['weight'] = log_W[e]  # set W(e) to actual weight
            for s in S:
                T_0.edges[s]['weight'] = log_W[s]  # set W(s) to actual weight

            log_W_1 = copy.deepcopy(log_S)  # guarantee S included
            log_W_1[e] = -torch.inf  # guarantee e excluded
            T_1: nx.DiGraph = get_start_arborescence(log_W_1, log_W_root, alg="edmonds")
            for s in S:
                T_1.edges[s]['weight'] = log_W[s]  # set W(s) to actual weight

            log_T0 = T_0.size(weight="weight")
            log_T1 = T_1.size(weight="weight")
            log_sum_T0_T1 = torch.logaddexp(log_T0, log_T1)
            theta = torch.exp(log_T0 - log_sum_T0_T1)
            U = torch.rand(1)
            if theta > U:
                # choose e
                S.append(e)
                log_S[e] = torch.inf  # guarantee e included
            else:
                continue


def sample_arborescence_old(log_W: torch.Tensor, log_W_root: torch.Tensor):
    logger = logging.getLogger('sample_arborescence')
    n_nodes = log_W.shape[0]
    T_init: nx.DiGraph = get_start_arborescence(log_W, log_W_root, alg="edmonds")
    T = T_init
    M = []

    idx_0 = get_ordered_indexes(n_nodes)
    idx_1 = get_ordered_indexes(n_nodes)

    log_T = 0

    # Based on old Slantis, might be completely wrong...
    for e0 in idx_0:

        # Case e in T
        if e0 in T.edges:
            # Create T_0, T_1 = cut(T, e)
            # Remove the edge from S, create a cut.
            T_copy = T.copy()
            T_copy.remove_edge(*e0)
            # Identify the components
            components = []
            for comp in nx.weakly_connected_components(T_copy):
                components.append(list(comp))

            T_0 = components[0]
            T_1 = components[1]
            # idx
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
            continue
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
                    # self.log_importance_weight += 0
                    # print("\tEdge ", edge, " is in S, we chose original S (no alternatives).\tp_keep: ", 1)

                # If edge was in the original S and we chose the alternative S
                elif accept_alt:
                    T = T_alternative
                    # print("\tEdge ", edge, " is in S, we chose alternative S.\tp_keep: ", p_keep)

                # If edge was in the original T and we chose the original T
                else:
                    M.append(e0)
                    log_T += log_p_not_accept
                    # print("\tEdge ", edge, " is in S, we chose original S.\tp_keep: ", p_keep)
            else:
                # If edge was in the alternative S and we chose the alternative S
                if not accept_alt:
                    T = T_alternative
                    # print("\tEdge ", edge, " is not in S, we chose alternative S.\tp_keep: ", p_keep)
                # If edge was in the alternative S and we chose the original S. Nothing changes.
                else:
                    # print("\tEdge ", edge, " is not in S, we chose original S.\tp_keep: ", p_keep)
                    pass

    return T, log_T
