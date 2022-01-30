import nngt
import networkx
import igraph as ig
import numpy as np

import network_symmetry as ns
from scipy import stats


def dir2undir(graph, w_label):
    graph_undir = ig.Graph()
    graph_undir.add_vertices(graph.vcount())
    for attr in graph.vs.attributes():
        graph_undir.vs[attr] = graph.vs[attr]

    edges = []
    weight = []
    for e in graph.es:
        v1, v2 = e.tuple
        edges.append((v1, v2))
        if w_label in graph.es.attributes():
            weight.append(e[w_label])

    graph_undir.add_edges(edges)
    if w_label in graph.es.attributes():
        graph_undir.es[w_label] = weight

    return graph_undir


# similaridade
def degree(graph_data, prop):
    graph = graph_data['net']
    w_label = graph_data['w_label']

    graph_simply = dir2undir(graph, w_label).simplify(multiple=True, loops=False,
                                                      combine_edges=graph_data['simply_method2'])
    if prop == 'nwnd':
        # sem peso e sem direção
        undir_unweig = graph_simply.degree()
        return undir_unweig

    # com peso
    if prop == 'yw':
        weig = graph_simply.strength(weights=w_label)
        return weig

    # com direção
    if prop == 'yd':
        in_degree = graph.degree(mode=ig.IN)
        return in_degree

    # com peso e com direção
    if prop == 'ywyd':
        in_strength = graph.strength(mode=ig.IN, weights=w_label)
        return in_strength


# similaridade
def evcent(graph_data, prop):
    graph = graph_data['net']
    w_label = graph_data['w_label']

    # sem peso e sem direção
    graph_simply = dir2undir(graph, w_label).simplify(multiple=True, loops=False,
                                                      combine_edges=graph_data['simply_method2'])
    if prop == 'nwnd':
        undir_unweig = graph_simply.evcent(directed=False)
        return undir_unweig

    # com peso
    if prop == 'yw':
        weig = graph_simply.evcent(directed=False, weights=w_label)
        return weig

    # com direção
    if prop == 'yd':
        in_evcent = graph.evcent(directed=True)
        return in_evcent

    # com peso e com direção
    if prop == 'ywyd':
        in_weig_evcent = graph.evcent(weights=w_label)
        return in_weig_evcent


def pagerank(graph_data, prop):
    graph = graph_data['net']
    w_label = graph_data['w_label']

    # sem peso e sem direção
    graph_simply = dir2undir(graph, w_label).simplify(multiple=True, loops=False,
                                                      combine_edges=graph_data['simply_method2'])
    if prop == 'nwnd':
        undir_unweig = graph_simply.pagerank(directed=False)
        return undir_unweig

    # com peso
    if prop == 'yw':
        weig = graph_simply.pagerank(directed=False, weights=w_label)
        return weig

    # com direção
    if prop == 'yd':
        in_evcent = graph.pagerank(directed=True)
        return  in_evcent

    # com peso e com direção
    if prop == 'ywyd':
        in_weig_evcent = graph.pagerank(directed=True, weights=w_label)
        return in_weig_evcent


# testar se tem diferença entre com direção e sem direção
def constraint(graph_data, prop):
    graph = graph_data['net']
    w_label = graph_data['w_label']

    # sem peso e sem direção
    graph_simply = dir2undir(graph, w_label).simplify(multiple=True, loops=False,
                                                      combine_edges=graph_data['simply_method2'])

    if prop == 'nwnd':
        undir_unweig = graph_simply.constraint()
        return undir_unweig

    # com peso
    if prop == 'yw':
        weig = graph_simply.constraint(weights=w_label)
        return weig

    # com direção
    if prop == 'yd':
        in_constrain = graph.constraint()
        return in_constrain

    # com peso e com direção
    if prop == 'ywyd':
        in_weig_constrain = graph.constraint(weights=w_label)
        return in_weig_constrain


# similaridade
def katz(graph_data):
    graph = graph_data['net']
    w_label = graph_data['w_label']

    # sem peso e sem direção
    graph_simply = dir2undir(graph, w_label).simplify(multiple=True, loops=False,
                                                     combine_edges=graph_data['simply_method2'])

    # sem peso e sem direção
    undir_unweig = sorted(networkx.katz_centrality(graph_simply, weight=None))
    undir_unweig = list(undir_unweig.values())

    # com peso
    weig = sorted(networkx.katz_centrality(graph_simply, weight=w_label))
    weig = list(weig.values())

    # com direção
    in_katz = sorted(networkx.katz_centrality(graph.to_networkx(), weight=None))
    in_katz = list(in_katz.values())

    # com peso e com direção
    in_weig_katz = sorted(networkx.katz_centrality(graph.to_networkx(), weight=w_label))
    in_weig_katz = list(in_weig_katz.values())

    return undir_unweig, weig, in_katz, in_weig_katz


# distancia
def betweenness(graph_data, prop):
    graph = graph_data['net']
    w_label = graph_data['w_label']
    # sem peso e sem direção
    graph_simply = dir2undir(graph, w_label).simplify(multiple=True, loops=False,
                                                     combine_edges=graph_data['simply_method2'])
    if prop == 'nwnd':
        undir_unweig = graph_simply.betweenness(directed=False, weights=None, cutoff=False)
        return undir_unweig

    # com peso
    if prop == 'yw':
        weig = graph_simply.betweenness(directed=False, weights=w_label)
        return weig

    # com direção
    if prop == 'yd':
        bet_dir = graph.betweenness(directed=True, weights=None)
        return bet_dir

    # com peso e com direção
    if prop == 'ywyd':
        bet_dir_w = graph.betweenness(directed=True, weights=w_label)
        return bet_dir_w


# distancia
def closeness(graph_data, prop):
    graph = graph_data['net']
    w_label = graph_data['w_label']

    # sem peso e sem direção
    graph_simply = dir2undir(graph, w_label).simplify(multiple=True, loops=False,
                                                     combine_edges=graph_data['simply_method2'])
    if prop == 'nwnd':
        undir_unweig = graph_simply.closeness()
        return undir_unweig

    # com peso
    if prop == 'yw':
        weig = graph.closeness(weights=w_label)
        return weig

    # com direção
    if prop == 'yd':
        clos_in = graph.closeness(mode=ig.IN)
        return clos_in

    # com peso e com direção
    if prop == 'ywyd':
        clos_in_w = graph.closeness(mode=ig.IN, weights=w_label)
        return clos_in_w


# distance
def get_flow_betweenness(graph_data):
    graph = graph_data['net']
    w_label = graph_data['w_label']

    # sem peso e sem direção
    graph_simply = dir2undir(graph, w_label).simplify(multiple=True, loops=False,
                                                     combine_edges=graph_data['simply_method2'])

    undir_unweig = networkx.current_flow_betweenness_centrality(graph_simply.to_networkx(), weight=None)

    # com peso
    weig = networkx.current_flow_betweenness_centrality(graph_simply.to_networkx(), weight=w_label)

    # com direção
    flow_in = networkx.current_flow_betweenness_centrality(graph.to_networkx(), weight=None)

    # com peso e com direção
    flow_in_w = networkx.current_flow_betweenness_centrality(graph.to_networkx(), weight=w_label)

    return undir_unweig, weig, flow_in, flow_in_w


def shortest_path(graph_data):
    graph = graph_data['net']
    w_label = graph_data['w_label']
    flatten = lambda t: [item for sublist in t for item in sublist]

    graph.shortest_paths_dijkstra()

    # sem peso e sem direção
    graph_simply = dir2undir(graph, w_label).simplify(multiple=True, loops=False,
                                                     combine_edges=graph_data['simply_method2'])

    undir_unweig = graph_simply.shortest_paths_dijkstra(weights=None, mode=ig.ALL)
    undir_unweig = flatten(undir_unweig)

    # com peso
    weig = graph_simply.shortest_paths_dijkstra(weights=w_label, mode=ig.ALL)
    weig = flatten(weig)

    # com direção
    path_out = graph.shortest_paths_dijkstra(weights=None, mode=ig.OUT)
    path_out = flatten(path_out)

    # com peso e com direção
    path_out_w = graph.shortest_paths_dijkstra(weights=w_label, mode=ig.OUT)
    path_out_w = flatten(path_out_w)

    return undir_unweig, weig, path_out, path_out_w


def clustering_coef(graph_data, prop):
    graph = graph_data['net']
    w_label = graph_data['w_label']

    # sem peso e sem direção
    graph_simply = dir2undir(graph, w_label).simplify(multiple=True, loops=False,
                                                      combine_edges=graph_data['simply_method2'])

    graph_simply_nngt = nngt.Graph.from_library(graph_simply.to_networkx())
    if prop == 'nwnd':
        undir_unweig = nngt.analysis.local_clustering_binary_undirected(graph_simply_nngt)
        return undir_unweig

    if prop == 'yw':
        weig = nngt.analysis.local_clustering(graph_simply_nngt, directed=False, weights=w_label)
        return weig

    graph_nngt = nngt.Graph.from_library(graph.to_networkx())

    if prop == 'yd':
        c_coef_dir = nngt.analysis.local_clustering(graph_nngt, directed=True)
        return c_coef_dir

    if prop == 'ywyd':
        c_oef_dir_w = nngt.analysis.local_clustering(graph_nngt, directed=True, weights=w_label)
        return c_oef_dir_w


def triangle(graph_data, prop):
    graph = graph_data['net']
    w_label = graph_data['w_label']

    # sem peso e sem direção
    graph_simply = dir2undir(graph, w_label).simplify(multiple=True, loops=False,
                                                      combine_edges=graph_data['simply_method2'])

    graph_simply_nngt = nngt.Graph.from_library(graph_simply.to_networkx())
    if prop == 'nwnd':
        undir_unweig = nngt.analysis.triangle_count(graph_simply_nngt)
        return undir_unweig

    if prop == 'yw':
        weig = nngt.analysis.triangle_count(graph_simply_nngt, directed=False, weights=w_label)
        return weig

    graph_nngt = nngt.Graph.from_library(graph.to_networkx())

    if prop == 'yd':
        tri = nngt.analysis.triangle_count(graph_nngt, directed=True)
        return tri

    if prop == 'ywyd':
        tri = nngt.analysis.triangle_count(graph_nngt, directed=True, weights=w_label)
        return tri


def small_world_propensity(graph_data, prop):
    graph = graph_data['net']
    w_label = graph_data['w_label']

    # sem peso e sem direção
    graph_simply = dir2undir(graph, w_label).simplify(multiple=True, loops=False,
                                                      combine_edges=graph_data['simply_method2'])

    graph_simply_nngt = nngt.Graph.from_library(graph_simply.to_networkx())
    if prop == 'nwnd':
        undir_unweig = nngt.analysis.small_world_propensity(graph_simply_nngt)
        return undir_unweig

    if prop == 'yw':
        weig = nngt.analysis.small_world_propensity(graph_simply_nngt, directed=False, weights=w_label)
        return weig

    graph_nngt = nngt.Graph.from_library(graph.to_networkx())

    if prop == 'yd':
        smal = nngt.analysis.small_world_propensity(graph_nngt, directed=True)
        return smal

    if prop == 'ywyd':
        smal = nngt.analysis.small_world_propensity(graph_nngt, directed=True, weights=w_label)
        return smal

# def get_katz(graph_data, prop):
#     graph = graph_data['net']
#     w_label = graph_data['w_label']
#
#     # sem peso e sem direção
#     graph_simply = dir2undir(graph, w_label).simplify(multiple=True, loops=False,
#                                                       combine_edges=graph_data['simply_method2'])
#
#     graph_simply_nngt = nngt.Graph.from_library(graph_simply.to_networkx())
#     if prop == 'nwnd':
#         undir_unweig = nngt.analysis.(graph_simply_nngt)
#         return undir_unweig
#
#     if prop == 'yw':
#         weig = nngt.analysis.small_world_propensity(graph_simply_nngt, directed=False, weights=w_label)
#         return weig
#
#     graph_nngt = nngt.Graph.from_library(graph.to_networkx())
#
#     if prop == 'yd':
#         smal = nngt.analysis.small_world_propensity(graph_nngt, directed=True)
#         return smal
#
#     if prop == 'ywyd':
#         smal = nngt.analysis.small_world_propensity(graph_nngt, directed=True, weights=w_label)
#         return smal


def Score(X, num_obj_cat):
    """
    IN:
    X: [N,M] array, where M is the number of features and N the number of objects.
    num_obj_cat: array containing number of objects for each class

    OUT:
    Tc: Scatter distance
    """

    [numSamples, Dim] = X.shape

    numCat = num_obj_cat.size

    u = np.mean(X, axis=0)
    s = np.std(X, axis=0, ddof=1)

    B = X - u
    Z = B / s

    ind = np.cumsum(num_obj_cat)
    ind = np.concatenate(([0], ind))

    uCat = np.zeros([numCat, Dim])
    for k in range(numCat):
        data_class = Z[ind[k]:ind[k + 1]]
        uCat[k] = np.mean(data_class, axis=0)

    X_aux = Z.copy()
    for k in range(numCat):
        X_aux[ind[k]:ind[k + 1]] -= uCat[k]

    Sw = np.zeros([Dim, Dim])  # Within-cluster scatter matrix
    Sb = np.zeros([Dim, Dim])  # Between-cluster scatter matrix
    for k in range(numCat):
        data_class = X_aux[ind[k]:ind[k + 1]]

        Sw += np.dot(data_class.T, data_class)

        aux = (uCat[k] - 0.).reshape([1, Dim])
        Sb += num_obj_cat[k] * np.dot(aux.T, aux)

    C = np.dot(np.linalg.inv(Sw), Sb)
    Tc = np.trace(C)

    return Tc


def symmetry_values(graph_data, prop):
    graph = graph_data['net']
    w_label = graph_data['w_label']
    vertex_count = graph.vcount()
    edges = graph.get_edgelist()
    weights = graph.es[w_label]
    if prop == 'nwnd':
        measurer = ns.Network(vertex_count=vertex_count,
                              edges=edges,
                              directed=False,
                              weights=None)
    elif prop == 'yw':
        measurer = ns.Network(vertex_count=vertex_count,
                              edges=edges,
                              directed=False,
                              weights=weights)
    elif prop == 'yd':
        measurer = ns.Network(vertex_count=vertex_count,
                              edges=edges,
                              directed=True,
                              weights=None)
    elif prop == 'ywyd':
        measurer = ns.Network(vertex_count=vertex_count,
                              edges=edges,
                              directed=True,
                              weights=weights)

    else:
        print('not defined')
        return False

    measurer.set_parameters(h_max=2,
                            merge_last_level=True,
                            live_stream=False,
                            parallel_jobs=1,
                            verbose=False,
                            show_status=False
                            )

    measurer.compute_symmetry()
    h = 2
    accessibility = measurer.accessibility(h)
    symmetry_backbone = measurer.symmetry_backbone(h)
    symmetry_merged = measurer.symmetry_merged(h)
    return symmetry_backbone, symmetry_merged