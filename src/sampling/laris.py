"""
Code related to the Labeled Arborescence Importance Sampling (LArIS).
"""

import copy
import logging
import random

import networkx as nx
import numpy as np
import torch

import matplotlib
from networkx import maximum_spanning_arborescence

import matplotlib.pyplot as plt

from networkx.drawing.nx_pydot import graphviz_layout

matplotlib.use("Agg")  # to avoid interactive plots


def new_graph_force_arc(u, v, graph: nx.DiGraph) -> nx.DiGraph:
    # remove all incoming arcs for v except u,v
    arcs_to_v_no_u = [(a, b) for a, b in graph.in_edges(v) if a != u]
    new_graph = copy.deepcopy(graph)
    new_graph.remove_edges_from(arcs_to_v_no_u)
    return new_graph


def new_graph_with_arcs(ebunch, graph: nx.DiGraph) -> nx.DiGraph:
    new_graph = copy.deepcopy(graph)
    for u, v in ebunch:
        # remove all incoming arcs for v except u,v
        arcs_to_v_no_u = [(a, b) for a, b in graph.in_edges(v) if a != u]
        new_graph.remove_edges_from(arcs_to_v_no_u)
    return new_graph


def new_graph_without_arc(u, v, graph: nx.DiGraph) -> nx.DiGraph:
    new_graph = copy.deepcopy(graph)
    new_graph.remove_edge(u, v)
    return new_graph


def draw_graph(G: nx.DiGraph, to_file=None):
    pos = graphviz_layout(G, prog="dot")

    f: plt.figure.Figure = plt.figure(figsize=(5, 5))
    nx.draw(G, pos=pos, with_labels=True, ax=f.add_subplot(111))
    if to_file is not None:
        f.savefig(to_file, format="png")
    plt.show()


def sample_arborescence_from_weighted_graph(graph: nx.DiGraph,
                                            root: int = 0, debug: bool = False, order_method='random', temp=1.):
    # TODO: rename to laris
    # start with empty graph (only root)
    s = nx.DiGraph()
    s.add_node(root)
    # copy graph so to remove arcs which shouldn't be considered
    # while S gets constructed
    tempered_graph = copy.deepcopy(graph)
    if temp != 1.:
        for e in tempered_graph.edges:
            tempered_graph.edges[e]['weight'] = tempered_graph.edges[e]['weight'] * (1/temp)
    skimmed_graph = copy.deepcopy(graph)
    log_W = torch.tensor(nx.to_numpy_array(skimmed_graph)) * (1 / temp)
    # counts how many times arborescences cannot be found
    miss_counter = 0
    log_g = 0.
    candidate_arcs = get_ordered_arcs(skimmed_graph, method=order_method)

    while s.number_of_edges() < graph.number_of_nodes() - 1:
        # new graph with all s arcs
        g_with_s = new_graph_with_arcs(s.edges, tempered_graph)
        num_candidates_left = len(candidate_arcs)

        feasible_arcs = []
        for u, v in candidate_arcs:

            g_w = new_graph_force_arc(u, v, g_with_s)
            g_wo = new_graph_without_arc(u, v, g_with_s)
            t_w = t_wo = nx.DiGraph()  # empty graph
            try:
                t_w = maximum_spanning_arborescence(g_w, preserve_attrs=True)
                # save max feasible arcs
                feasible_arcs.append((u, v, tempered_graph.edges[u, v]['weight']))
                t_wo = maximum_spanning_arborescence(g_wo, preserve_attrs=True)
            except nx.NetworkXException as nxe:
                # go to next arc if, once some arcs are removed, no spanning arborescence exists
                miss_counter += 1
                if miss_counter in [100, 1000, 2000, 10000]:
                    logging.log(logging.WARNING, f'LArIS num misses: {miss_counter}')
                num_candidates_left -= 1
                if num_candidates_left > 0:
                    continue

            if num_candidates_left == 0 and len(feasible_arcs) > 0:
                # no arc allows for both t_w and t_wo to exist
                # must choose one of the feasible ones (for which t_w exists)
                # obliged choice -> theta = 1
                # theta = torch.tensor(1.)
                # randomize selection based on weights
                (u, v), theta = _sample_feasible_arc(feasible_arcs)
            elif num_candidates_left == 0:
                # heuristic: reset s
                logging.debug("No more candidates in LArIS tree reconstruction. Restarting algorithm.")
                s = nx.DiGraph()
                s.add_node(root)
                # skimmed_graph = copy.deepcopy(graph)
                break
            else:
                if t_w.number_of_nodes() == 0 or t_wo.number_of_nodes() == 0:
                    raise Exception('t_w and t_wo are empty but being called')
                w_Tw = torch.tensor([log_W[u, v] for (u, v) in t_w.edges()]).sum()
                w_To = torch.tensor([log_W[u, v] for (u, v) in t_wo.edges()]).sum()
                theta = torch.exp(w_Tw - torch.logaddexp(w_Tw, w_To))
                # theta2 = torch.exp(t_w.size(weight='weight') -
                #                   torch.logaddexp(t_w.size(weight='weight'), t_wo.size(weight='weight')))

            if torch.rand(1) < theta:
                s.add_edge(u, v, weight=graph.edges[u, v]['weight'])
                # remove all incoming arcs to v (including u,v)
                # skimmed_graph.remove_edges_from(graph.in_edges(v))
                # skimmed_graph.remove_edges_from([(v, u)])
                candidates_to_remove = list(graph.in_edges(v))
                candidates_to_remove.append((v, u))
                candidate_arcs = [a for a in candidate_arcs if a not in candidates_to_remove]
                # prob of sampling the tree: prod of bernoulli trials
                log_g += torch.log(theta)
                # go to while and check if s is complete
                break

    return s, log_g


def _sample_feasible_arc(weighted_arcs):
    # weighted_arcs is a list of 3-tuples (u, v, weight)
    # weights are negative: need transformation
    unnorm_probs = 1 / (-torch.stack([w for u, v, w in weighted_arcs]))
    probs = unnorm_probs / unnorm_probs.sum()
    c = np.random.choice(np.arange(len(weighted_arcs)), p=probs.numpy())
    return weighted_arcs[c][:2], probs[c]


def get_ordered_arcs(graph: nx.DiGraph, method='random'):
    edges_list = list(graph.edges)
    if method == 'random':
        order = np.random.permutation(len(edges_list))
        ordered_edges = []
        for i in order:
            ordered_edges.append(edges_list[i])
    elif method == 'edmonds':
        mst_graph = nx.maximum_spanning_arborescence(graph).edges
        mst_arcs = list(mst_graph)
        graph.remove_edges_from(mst_arcs)
        ordered_edges = mst_arcs + list(graph.edges)
    else:
        raise ValueError(f'Method {method} is not available.')

    return ordered_edges


def sample_rand_mst(weighted_graph: nx.DiGraph) -> (nx.DiGraph, float):
    # sample an arc
    u, v = random.sample(weighted_graph.edges, 1)[0]
    graph_with_arc = new_graph_force_arc(u, v, weighted_graph)
    try:
        mst = maximum_spanning_arborescence(graph_with_arc, preserve_attrs=True)
    except nx.NetworkXException as nex:
        mst = maximum_spanning_arborescence(weighted_graph)
    log_w = mst.size()
    return mst, log_w
