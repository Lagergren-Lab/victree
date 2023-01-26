import copy
import logging
import random
from typing import Tuple, List

import networkx as nx
import numpy as np
import torch

import matplotlib
from networkx import maximum_spanning_arborescence

matplotlib.use("Agg") # to avoid interactive plots
import matplotlib.pyplot as plt

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


def new_graph_with_arc(u, v, graph: nx.DiGraph) -> nx.DiGraph:
    # remove all incoming arcs for v except u,v
    arcs_to_v_no_u = [(a, b) for a, b in graph.in_edges(v) if a != u]
    new_graph = nx.DiGraph.copy(graph)
    new_graph.remove_edges_from(arcs_to_v_no_u)
    return new_graph


def new_graph_with_arcs(ebunch, graph: nx.DiGraph) -> nx.DiGraph:
    new_graph = nx.DiGraph.copy(graph)
    for u, v in ebunch:
        # remove all incoming arcs for v except u,v
        arcs_to_v_no_u = [(a, b) for a, b in graph.in_edges(v) if a != u]
        new_graph.remove_edges_from(arcs_to_v_no_u)
    return new_graph


def new_graph_without_arc(u, v, graph: nx.DiGraph) -> nx.DiGraph:
    new_graph = nx.DiGraph.copy(graph)
    new_graph.remove_edge(u, v)
    return new_graph


def get_ordered_indexes(n_nodes):
    idx = []
    for e1 in range(0, n_nodes):
        for e2 in range(1, n_nodes):  # exclude node to root case
            if e1 != e2: idx.append((e1, e2))

    return idx


def check_cycles(T_copy):
    pass


def draw_graph(G: nx.DiGraph, to_file=None):
    pos = graphviz_layout(G, prog="dot")

    f: plt.figure.Figure = plt.figure(figsize=(5,5))
    nx.draw(G, pos=pos, with_labels=True, ax=f.add_subplot(111))
    if to_file is not None:
        f.savefig(to_file, format="png")
    plt.show()


def is_root(param):
    pass


def sample_arborescence_from_weighted_graph(graph: nx.DiGraph,
                                   root: int = 0, debug: bool = False):
    # start with empty graph (only root)
    s = nx.DiGraph()
    s.add_node(root)
    # copy graph so to remove arcs which shouldn't be considered
    # while S gets constructed
    skimmed_graph = nx.DiGraph.copy(graph)
    # counts how many times arborescences cannot be found
    miss_counter = 0
    log_isw = 0.
    while s.number_of_edges() < graph.number_of_nodes() - 1:
        candidate_arcs = get_ordered_arcs(skimmed_graph.edges)
        # new graph with all s arcs
        g_with_s = new_graph_with_arcs(s.edges, graph)
        num_candidates_left = len(candidate_arcs)

        feasible_arcs = []
        for u, v in candidate_arcs:

            g_w = new_graph_with_arc(u, v, g_with_s)
            g_wo = new_graph_without_arc(u, v, g_with_s)
            t_w = t_wo = nx.DiGraph()  # empty graph
            try:
                t_w = maximum_spanning_arborescence(g_w, preserve_attrs=True)
                # save max feasible arcs
                feasible_arcs.append((u, v, graph.edges[u, v]['weight']))
                t_wo = maximum_spanning_arborescence(g_wo, preserve_attrs=True)
            except nx.NetworkXException as nxe:
                # go to next arc if, once some arcs are removed, no spanning arborescence exists
                miss_counter += 1
                if miss_counter in [100, 1000, 2000, 10000]:
                    logging.log(logging.WARNING, f'DSlantis num misses: {miss_counter}')
                num_candidates_left -= 1
                if num_candidates_left > 0:
                    continue

            if num_candidates_left == 0 and len(feasible_arcs) > 0:
                # no arc allows for both t_w and t_wo to exist
                # must choose one of the feasible ones (for which t_w exists)
                # obliged choice -> theta = 1
                theta = 1.
                # randomize selection based on weights
                u, v = _sample_feasible_arc(feasible_arcs)
            elif num_candidates_left == 0:
                # heuristic: reset s
                s = nx.DiGraph()
                s.add_node(root)
                skimmed_graph = nx.DiGraph.copy(graph)
                break
            else:
                if t_w.number_of_nodes() == 0 or t_wo.number_of_nodes() == 0:
                    raise Exception('t_w and t_wo are empty but being called')
                theta = t_w.size(weight='weight') / (t_w.size(weight='weight') + t_wo.size(weight='weight'))

            if np.random.rand() < theta:
                s.add_edge(u, v, weight=graph.edges[u, v]['weight'])
                # remove all incoming arcs to v (including u,v)
                skimmed_graph.remove_edges_from(graph.in_edges(v))
                skimmed_graph.remove_edges_from([(v, u)])
                log_isw += np.log(theta)
                # go to while and check if s is complete
                break

    return s, log_isw


def _sample_feasible_arc(weighted_arcs):
    # weighted_arcs is a list of 3-tuples (u, v, weight)
    unnorm_probs = np.array([w for u, v, w in weighted_arcs])
    probs = unnorm_probs / unnorm_probs.sum()
    c = np.random.choice(np.arange(len(weighted_arcs)), p=probs)
    return weighted_arcs[c][:2]


def get_ordered_arcs(edges, method='random'):
    ordered_edges = []
    edges_list: list
    if isinstance(edges, list):
        edges_list = edges
    else:  # OutEdgeView
        edges_list = [(u, v) for u, v in edges]
    if method == 'random':
        order = np.random.permutation(len(edges))
    else:
        raise ValueError(f'Method {method} is not available.')

    for i in order:
        ordered_edges.append(edges_list[i])
    return ordered_edges


def sample_arborescence(log_W: torch.Tensor, 
                        root: int,
                        debug=False) -> Tuple[nx.DiGraph, torch.Tensor]:
    logger = logging.getLogger('sample_arborescence')
    n_nodes = log_W.shape[0]
    S = []  # Selected arcs
    S_nodes = set(())
    roots = set(())
    children = set(())
    log_S = torch.tensor(0.)
    S_arborescence = nx.DiGraph()
    log_W_with_S = copy.deepcopy(log_W)
    including_weight = torch.max(log_W) + torch.log(torch.tensor(n_nodes))

    idx_0 = get_ordered_indexes(n_nodes)  # Arc proposal set
    random.shuffle(idx_0)

    n_iterations = 0
    while len(S) < n_nodes - 1 or n_iterations > 100:
        n_iterations += 1
        for a in idx_0:
            # Create comparison tree T_0 with a and T_1 without 'a'. Both including S set.
            # T_0
            log_W_0 = copy.deepcopy(log_W_with_S)       # guarantee S included
            log_W_0[a] = including_weight               # guarantee 'a' included
            T_0 = get_edmonds_arborescence(log_W_0, root)
            T_0.edges[a]['weight'] = log_W[a]           # set W(a) to actual weight

            # T_1
            log_W_1 = copy.deepcopy(log_W_with_S)   # guarantee S included
            log_W_1[a] = -torch.inf                 # guarantee 'a' excluded
            T_1: nx.DiGraph = get_edmonds_arborescence(log_W_1, root)
            if debug:   # TODO: make debug system professional
                assert set(S) <= set(T_0.edges)
                assert a in set(T_0.edges)
                assert set(S) <= set(T_1.edges)
                assert a not in set(T_1.edges)

            for s in S:
                T_0.edges[s]['weight'] = log_W[s]   # reset weight of arc T_0(s) to actual weight
                T_1.edges[s]['weight'] = log_W[s]   # reset weight of arc T_1(s) to actual weight

            # Create selection probability theta
            log_T0 = T_0.size(weight="weight")
            log_T1 = T_1.size(weight="weight")
            log_sum_T0_T1 = torch.logaddexp(log_T0, log_T1)
            theta = torch.exp(log_T0 - log_sum_T0_T1)

            # Bernoulli trial
            U = torch.rand(1)
            if theta > U:
                # select a
                S.append(a)
                u, v = a
                children.add(v)
                if u not in children:
                    roots.add(u)
                roots.discard(v)

                S_nodes.add(u)
                S_nodes.add(v)  # only used for debugging. Can be removed.
                S_arborescence.add_edge(u, v)

                log_S += torch.log(theta)
                log_W_with_S[a] = including_weight  # guarantee 'a' included
                cycles_inducing_arcs = [(v, s) for s in children if s != v]  # filter out possible cycles

                # filter out children to root connections in sub_arboresence containing e.
                sub_arboresences = nx.weakly_connected_components(S_arborescence)
                for sub_arb in sub_arboresences:
                    if v in sub_arb:
                        self_root = roots.intersection(sub_arb).pop()
                        for child in children.intersection(sub_arb):
                            cycles_inducing_arcs.append((child, self_root))

                for a_cycle in cycles_inducing_arcs:
                    if a_cycle in idx_0:    # replace if statements with faster system
                        idx_0.remove(a_cycle)
                    if (a_cycle[1], a_cycle[0]) in idx_0:
                        idx_0.remove((a_cycle[1], a_cycle[0]))
                idx_0[:] = [x for x in idx_0 if x[1] != v]  # filter out possible co-parent arcs

            else:
                continue

    if debug:
        logger.debug(f"Number of proposal iterations: {n_iterations}")
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
