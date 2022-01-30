import os
import glob
import json
import nngt
import zipfile
import networkx
import numpy as np
import pandas as pd
import igraph as ig
import multiprocessing
import sys
import powerlaw
import xnetwork as xn
from sklearn.preprocessing import minmax_scale

import matplotlib.pyplot as plt
from copy import deepcopy
from functools import partial


def get_model_dir(model, model_args, reciprocity, edge_distribution, w_distribution):
    g = model(**model_args)
    giant = g.components().giant()

    g_directed = ig.Graph(directed=True)
    g_directed.add_vertices(giant.vcount())
    edges = []

    edge_reciprocity = edge_distribution[0](*edge_distribution[1], giant.ecount())
    quantile = np.percentile(edge_reciprocity, 100 * reciprocity)
    print(quantile)
    dists = []

    # usar percentil como corte

    for e, r_random in zip(giant.es, edge_reciprocity):
        v1, v2 = e.tuple
        d = 0
        if 'dist' in giant.es.attributes():
            d = e['dist']
        if np.random.uniform(0, 1) > 0.5:
            edges.append((v1, v2))
            dists.append(d)
            if r_random > quantile:
                edges.append((v2, v1))
                dists.append(d)
        else:
            edges.append((v2, v1))
            dists.append(d)
            if r_random > reciprocity:
                edges.append((v1, v2))
                dists.append(d)
    g_directed.add_edges(edges)

    if 'dist' in giant.es.attributes():
        g_directed.es['distance'] = dists

    print("%.2f %.2f %.2f %d %d" %
          (1 - reciprocity, np.mean(giant.degree()), np.mean(g_directed.degree()),
           g_directed.vcount(), g_directed.ecount()))
    return g_directed


def get_model_w(model, model_args, threshold, w_distribution):
    g = model(**model_args)
    giant = g.components().giant()

    g_w = ig.Graph()
    g_w.add_vertices(giant.vcount())
    edges = []

    W = w_distribution[0](*w_distribution[1], giant.ecount())

    dists = []
    percentil = np.percentile(W, threshold * 100)
    for e, w in zip(giant.es, W):
        v1, v2 = e.tuple
        if w > percentil:
            edges.append((v2, v1))
            dists.append(w)
    g_w.add_edges(edges)

    if 'dist' in giant.es.attributes():
        g_w.es['distance'] = dists
    else:
        g_w.es['similarity'] = dists

    print("%s %.2f %.2f %d %d" %
          (model.__name__, threshold, np.mean(giant.degree()), g_w.vcount(), g_w.ecount()))
    return g_w


def normal(a, b, size=None):
    x = np.random.normal(loc=0, scale=1, size=size)
    min_x = min(x)
    max_x = max(x)

    return (b - a) * (x - min_x) / (max_x - min_x) + a


def waxman(n, beta, alpha):
    netx = networkx.generators.geometric.waxman_graph(n, beta, alpha)
    igra = ig.Graph.from_networkx(netx)
    dists = []
    for e in igra.es:
        v1, v2 = e.tuple
        p1 = igra.vs[v1]['pos']
        p2 = igra.vs[v2]['pos']
        d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        dists.append(d)
    igra.es['dist'] = dists
    return igra


def random_geometric(n, radius):
    netx = networkx.generators.geometric.random_geometric_graph(n, radius)
    igra = ig.Graph.from_networkx(netx)
    for e in igra.es:
        v1, v2 = e.tuple
        p1 = igra.vs[v1]['pos']
        p2 = igra.vs[v2]['pos']
        e['dist'] = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return igra


def random_partition_graph(n, c, p_in, p_out):
    comms = [int(n / c) - 1 for _ in range(c)]
    netx = networkx.generators.community.random_partition_graph(comms, p_in, p_out)
    igra = ig.Graph.from_networkx(netx)

    return igra


def get_models_dir(n_vertices, n_r, samples=3):
    models = [
        (random_partition_graph, {'n': n_vertices, 'c': 4, 'p_in': 0.3, 'p_out': 0.03}),
        (waxman, {'n': n_vertices, 'beta': 0.4, 'alpha': 0.14}),
        (random_geometric, {'n': n_vertices, 'radius': 0.13}),
        (ig.Graph.Erdos_Renyi, {'n': n_vertices, 'p': 0.05}),  # mudar pra arestas ao invés de p
        (ig.Graph.Barabasi, {'n': n_vertices, 'm': 12}),
        (ig.Graph.Watts_Strogatz, {'dim': 2, 'size': 23, 'nei': 5, 'p': 0.0001}),  # passar para dim 1
        (ig.Graph.Watts_Strogatz, {'dim': 2, 'size': 23, 'nei': 5, 'p': 0.002}),  # nei = 3?????
        (ig.Graph.Watts_Strogatz, {'dim': 2, 'size': 23, 'nei': 5, 'p': 1}),
    ]

    idx = 0
    alpha = 3
    plaw = lambda a, b, n: 1 - (
            (b ** (alpha + 1) - a ** (alpha + 1)) * np.random.uniform(0, 1, n) + a ** (alpha + 1)) ** (
                                   1 / (alpha + 1))
    nets = dict()

    edge_dists = {'random': (np.random.uniform, (0, 1)),
                  'weibull': (np.random.weibull, (5,)),
                  'powerlaw': (plaw, (0.1, 1))}

    for model in models:
        m_name = model[0].__name__
        print(model[0].__name__)
        for dist_name, edge_dist in edge_dists.items():
            for _ in range(samples):
                for r in np.linspace(0, 1, n_r):
                    g = get_model_dir(*model, r, edge_dist, None)
                    giant = g.components().giant()
                    name = m_name + '_' + str(idx)
                    idx += 1
                    print(name, 1 - r, giant.reciprocity(mode='ratio'), np.mean(giant.degree()))
                    nets[name] = {'model': m_name + ('(p=%.4f)' % model[1]['p'] if 'Watts' in m_name else ''),
                                  'w_dist': dist_name,
                                  'r': 1 - r,
                                  'w_label': 'distance' if 'distance' in giant.es.attributes() else 'similarity',
                                  'simply_method1': 'mean',
                                  'simply_method2': 'mean',
                                  'net': giant
                                  }

            print()
    return nets


def plaw(a, b, n):
    alpha = 3
    return 1 - ((b ** (alpha + 1) - a ** (alpha + 1)) * np.random.uniform(0, 1, n) + a ** (alpha + 1)) ** (
                                       1 / (alpha + 1))


def get_models_w(n_vertices, n_r, samples=3):
    models = [
        (random_partition_graph, {'n': n_vertices, 'c': 4, 'p_in': 0.3, 'p_out': 0.03}),
        (waxman, {'n': n_vertices, 'beta': 0.4, 'alpha': 0.14}),
        (random_geometric, {'n': n_vertices, 'radius': 0.13}),
        (ig.Graph.Erdos_Renyi, {'n': n_vertices, 'p': 0.05}), # mudar pra arestas ao invés de p
        (ig.Graph.Barabasi, {'n': n_vertices, 'm': 12}),
        (ig.Graph.Watts_Strogatz, {'dim': 2, 'size': 23, 'nei': 5, 'p': 0.0001}), # passar para dim 1
        (ig.Graph.Watts_Strogatz, {'dim': 2, 'size': 23, 'nei': 5, 'p': 0.002}), # nei = 3?????
        (ig.Graph.Watts_Strogatz, {'dim': 2, 'size': 23, 'nei': 5, 'p': 1}),
    ]

    idx = 0
    nets = dict()

    w_dists = {'random': (np.random.uniform, (0, 1)),
               'weibull': (np.random.weibull, (5,)),
               'powerlaw': (plaw, (0.1, 1))}

    for model in models:
        m_name = model[0].__name__
        print(model[0].__name__)
        for w_name, w_dist in w_dists.items():
            print(w_name)
            for _ in range(samples):
                for t in np.linspace(0, 0.99, n_r):
                    g = get_model_w(*model, t, w_dist)
                    giant = g.components().giant()
                    name = m_name + '_' + str(idx)
                    idx += 1
                    nets[name] = {'model': m_name + ('(p=%.4f)' % model[1]['p'] if 'Watts' in m_name else ''),
                                  'w_dist': w_name,
                                  'r': 1 - t,
                                  'w_label': 'distance' if 'distance' in giant.es.attributes() else 'similarity',
                                  'simply_method1': 'mean',
                                  'simply_method2': 'mean',
                                  'net': giant
                                  }

                print()
    return nets


def get_xnet_str(g, fd, ignoredNodeAtts=[], ignoredEdgeAtts=[]):
    N = g.vcount()
    E = g.ecount()

    nodesAtt = g.vs.attributes()
    edgesAtt = g.es.attributes()

    if ('weight' in nodesAtt) and ('weight' not in ignoredNodeAtts):
        isNodeWeighted = True
        isNodeWeightedString = 'weighted'
    else:
        isNodeWeighted = False
        isNodeWeightedString = 'nonweighted'

    if ('name' in nodesAtt) and ('name' not in ignoredNodeAtts):
        isNodeNamed = True
    else:
        isNodeNamed = False


    fd.write('#vertices ' + str(N) + ' ' + isNodeWeightedString + '\n')

    if isNodeNamed and isNodeWeighted:
        for i in range(N):
            fd.write('\"' + g.vs[i]['name'] + '\" ' + str(g.vs[i]['weight']) + '\n')
    elif isNodeNamed and (not isNodeWeighted):
        for i in range(N):
            fd.write('\"' + g.vs[i]['name'] + '\"' + '\n')
    elif (not isNodeNamed) and isNodeWeighted:
        for i in range(N):
            fd.write(str(g.vs[i]['weight']) + '\n')

    if ('weight' in edgesAtt) and ('weight' not in ignoredEdgeAtts):
        isEdgeWeighted = True
        isEdgeWeightedString = 'weighted'
    else:
        isEdgeWeighted = False
        isEdgeWeightedString = 'nonweighted'

    if g.is_directed() == True:
        isEdgeDirected = True
        isEdgeDirectedString = 'directed'
    else:
        isEdgeDirected = False
        isEdgeDirectedString = 'undirected'

    fd.write('#edges ' + isEdgeWeightedString + ' ' + isEdgeDirectedString + '\n')

    for i in range(E):

        edge = g.es[i].tuple
        if isEdgeWeighted:
            fd.write(str(edge[0]) + ' ' + str(edge[1]) + ' ' + str(g.es[i]['weight']) + '\n')
        else:
            fd.write(str(edge[0]) + ' ' + str(edge[1]) + '\n')

    if isNodeWeighted:
        nodesAtt.remove('weight')
    if isNodeNamed:
        nodesAtt.remove('name')
    if isEdgeWeighted:
        edgesAtt.remove('weight')

    nodesAtt.sort()
    edgesAtt.sort()

    isPython2 = sys.version_info[0] == 2
    if isPython2:
        stringType = basestring
    else:
        stringType = str

    for att in nodesAtt:
        if att not in ignoredNodeAtts:
            sample = g.vs[0][att]

            typeSample = type(sample)
            if isinstance(sample, stringType):
                typeSampleString = 's'
            elif np.isscalar(sample) == True:
                typeSampleString = 'n'
            elif (typeSample == list) or (typeSample == tuple) or (typeSample == np.ndarray):
                if len(sample) == 2:
                    typeSampleString = 'v2'
                elif len(sample) == 3:
                    typeSampleString = 'v3'

            fd.write('#v \"' + att + '\" ' + typeSampleString + '\n')
            for i in range(N):
                if typeSampleString == 'n':
                    fd.write(str(g.vs[i][att]) + '\n')
                elif typeSampleString == 'v2':
                    fd.write(str(g.vs[i][att][0]) + ' ' + str(g.vs[i][att][1]) + '\n')
                elif typeSampleString == 'v3':
                    fd.write(str(g.vs[i][att][0]) + ' ' + str(g.vs[i][att][1]) + ' ' + str(g.vs[i][att][2]) + '\n')
                elif typeSampleString == 's':
                    fd.write('\"' + g.vs[i][att] + '\"' + '\n')

    for att in edgesAtt:
        if att not in ignoredEdgeAtts:
            sample = g.es[0][att]

            typeSample = type(sample)
            if isinstance(sample, stringType):
                typeSampleString = 's'
            elif np.isscalar(sample) == True:
                typeSampleString = 'n'
            elif (typeSample == list) or (typeSample == tuple) or (typeSample == np.ndarray):
                if len(sample) == 2:
                    typeSampleString = 'v2'
                elif len(sample) == 3:
                    typeSampleString = 'v3'

            fd.write('#e \"' + att + '\" ' + typeSampleString + '\n')
            for i in range(E):
                if typeSampleString == 'n':
                    fd.write(str(g.es[i][att]) + '\n')
                elif typeSampleString == 'v2':
                    fd.write(str(g.es[i][att][0]) + ' ' + str(g.es[i][att][1]) + '\n')
                elif typeSampleString == 'v3':
                    fd.write(str(g.es[i][att][0]) + ' ' + str(g.es[i][att][1]) + ' ' + str(g.es[i][att][2]) + '\n')
                elif typeSampleString == 's':
                    fd.write('\"' + g.es[i][att] + '\"' + '\n')


def save_models(models, folder):
    model_copy = deepcopy(models)
    for k, model in models.items():
        print(model)
        nets = []
        for key, instance in model.items():
            nets.append(instance['net'])
            model_copy[k][key]['net'] = ''
        handler = open('data/models/%s.xnet' % k, 'w')
        for i, net in enumerate(nets):
            get_xnet_str(net, handler)
            handler.write('\n-------\n')
        handler.close()

    with open('data/models/%s/models.json' % folder, 'w') as outfile:
        json.dump(model_copy, outfile)



def get_family(infos):
    if 'w_dist' in infos:
        return infos['model'] + ' ' + infos['w_dist']
    else:
        return infos['model']
    # return name.split(' ')[1] + ' ' + name.split(' ')[2]


def load_models(folder):
    jsonfile = glob.glob('data/models/%s/models.json' % folder)[0]
    with open(jsonfile, 'r') as inputfile:
        model_infos = json.load(inputfile)

    for key, infos in model_infos.items():
        net = ig.Graph().Read_GML('data/models/%s/%s.gml' % (folder, key))
        family = get_family(infos)
        infos['family'] = family
        infos['net'] = net
        infos['name'] = 'r=%.2f e=%d' % (infos['r'], net.ecount())

    return model_infos


def get_bases_infos():
    valid_nets = {
        'webkb_webkb_washington_link1': {
            'w_label': 'weight',
            'simply_method1': 'mean',
            'simply_method2': 'mean',
            'type': 'sim'
        },
        'residence_hall_residence_hall': {
            'w_label': 'weight',
            'simply_method1': 'mean',
            'simply_method2': 'mean',
            'type': 'sim'
        },
        'celegans_2019_hermaphrodite_chemical': {
            'w_label': 'connectivity',
            'simply_method1': 'mean',
            'simply_method2': 'mean',
            'type': 'sim'
        },
        'celegans_2019_hermaphrodite_chemical_corrected': {
            'w_label': 'connectivity',
            'simply_method1': 'mean',
            'simply_method2': 'mean',
            'type': 'sim'
        },
        'celegans_2019_hermaphrodite_chemical_synapse': {
            'w_label': 'synapses',
            'simply_method1': 'mean',
            'simply_method2': 'mean',
            'type': 'sim'
        },
        'copenhagen_calls': {
            'w_label': 'duration',
            'simply_method1': 'mean',
            'simply_method2': 'mean',
            'type': 'sim'
        },
        'advogato_advogato': {
            'w_label': 'weight',
            'simply_method1': 'mean',
            'simply_method2': 'mean',
            'type': 'sim'
        },  # SOCIAL similaridade
        'bitcoin_alpha_bitcoin_alpha': {
            'w_label': 'rating',
            'simply_method1': 'mean',
            'simply_method2': 'mean',
            'type': 'sim'
        },  # SOCIAL similaridade [-10,+10]
        'bitcoin_trust_bitcoin_trust': {
            'w_label': 'rating',
            'simply_method1': 'mean',
            'simply_method2': 'mean',
            'type': 'sim'
        },  # similaridade [-10,+10]
        'foldoc_foldoc': {
            'w_label': 'weight',
            'simply_method1': 'mean',
            'simply_method2': 'sum',
            'type': 'sim'
        },  # INFORMATIONAL
        # 'messal_shale_messal_shale': {
        #     'w_label': 'certainty',
        #     'simply_method1': 'mean',
        #     'simply_method2': 'mean',
        #     'type': 'sim'
        # },  # BIOLOGICAL [small]
        'us_agencies_california': {
            'w_label': 'link_counts',
            'simply_method1': 'sum',
            'simply_method2': 'sum',
            'type': 'sim'
        },  # POLITICAL
        'us_agencies_newyork': {
            'w_label': 'link_counts',
            'simply_method1': 'sum',
            'simply_method2': 'sum',
            'type': 'sim'
        },  # POLITICAL,
        'us_agencies_aggregate': {
            'w_label': 'link_counts',
            'simply_method1': 'sum',
            'simply_method2': 'sum',
            'type': 'sim'
        },  # POLITICAL
        'openflights_openflights': {
            'w_label': 'distance',
            'simply_method1': 'mean',
            'simply_method2': 'mean',
            'type': 'dist'
        }  # TRANSPORTATION
    }
    return valid_nets


def get_bases_graphs():
    files = glob.glob('data/bases/*.zip')

    valid = get_bases_infos()

    for file in valid:

        header = 'data/bases/' + file
        if not os.path.isdir(header):
            os.mkdir('data/bases/' + file)
        with zipfile.ZipFile(header + '.csv.zip', 'r') as zip_ref:
            zip_ref.extractall('data/bases/' + file)

        nodes = pd.read_csv(header + '/nodes.csv', escapechar='\\')
        edges = pd.read_csv(header + '/edges.csv')

        g = ig.Graph(directed=True)
        g.add_vertices(len(nodes))
        for col in nodes.columns:
            if not nodes[col].all() and 'index' not in col:
                g.vs[col] = nodes[col]

        sources = edges['# source'].values
        targets = edges[' target'].values
        g.add_edges(zip(sources, targets))

        w_label = valid[file]['w_label']

        for col in edges.columns:
            if w_label in col:
                g.es[w_label] = edges[col]

        if 'bitcoin' in header:
            g.es[w_label] = [w + 11 for w in g.es[w_label]]

        if 'copenhagen' in header:
            g.es[w_label] = [w + 2 for w in g.es[w_label]]

        if 'openflights' in header:
            g.es[w_label] = [w + 0.2 for w in g.es[w_label]]

        g.simplify(multiple=True, loops=False, combine_edges=valid[file]['simply_method1'])
        giant = g.clusters(mode='strong').giant()

        valid[file]['net'] = giant

    for name in valid:
        if not 'net' in valid[name]:
            del valid[name]

    return valid


def param_analysis():
    g = ig.Graph.Watts_Strogatz
    p = [{'dim': 1, 'size': 500, 'nei': 5, 'p': 0.0001},
         {'dim': 1, 'size': 500, 'nei': 5, 'p': 0.001},
         {'dim': 1, 'size': 500, 'nei': 5, 'p': 0.002},
         {'dim': 1, 'size': 500, 'nei': 5, 'p': 0.003},
         {'dim': 1, 'size': 500, 'nei': 5, 'p': 0.01},
         {'dim': 1, 'size': 500, 'nei': 5, 'p': 0.02},
         {'dim': 1, 'size': 500, 'nei': 5, 'p': 0.03},
         {'dim': 1, 'size': 500, 'nei': 5, 'p': 0.1},
         {'dim': 1, 'size': 500, 'nei': 5, 'p': 1}]
    X, Y, Z, W = [], [], [], []
    for ptemp in p:
        gtemp = g(**ptemp)
        nngtemp = nngt.Graph.from_library(gtemp.to_networkx())
        small, l, c = nngt.analysis.small_world_propensity(nngtemp, return_deviations=True)
        print('nngt small propensity=%.2f (l=%.2f c=%.2f)' % (small, l, c))
        X.append(ptemp['p'])
        Y.append(l)
        Z.append(c)
        W.append(small)

    plt.plot(X, Y, 'o', linestyle='dashed', label='delta l')
    plt.plot(X, Z, 'o', linestyle='dashed', label='delta c')
    plt.xscale('log')
    plt.legend()
    plt.title('deviation of C_obs and L_obs')
    plt.savefig('delta_l_delta_c.pdf')
    plt.show()

    plt.plot(X, W, 'o', linestyle='dashed')
    plt.xscale('log')
    plt.savefig('small-world-property.pdf')
    plt.show()


def get_model_w_dir(model, model_args, reciprocity, edge_distribution, w_distribution):
    g = model(**model_args)
    giant = g.components().giant()

    g_directed = ig.Graph(directed=True)
    g_directed.add_vertices(giant.vcount())
    edges = []

    edge_reciprocity = edge_distribution[0](*edge_distribution[1], giant.ecount())
    quantile = np.percentile(edge_reciprocity, 100 * reciprocity)
    print(quantile)
    dists = []

    # usar percentil como corte

    for e, r_random in zip(giant.es, edge_reciprocity):
        v1, v2 = e.tuple
        d = 0
        if 'dist' in giant.es.attributes():
            d = e['dist']
        if np.random.uniform(0, 1) > 0.5:
            edges.append((v1, v2))
            dists.append(d)
            if r_random > quantile:
                edges.append((v2, v1))
                dists.append(d)
        else:
            edges.append((v2, v1))
            dists.append(d)
            if r_random > reciprocity:
                edges.append((v1, v2))
                dists.append(d)
    g_directed.add_edges(edges)

    if 'dist' in giant.es.attributes():
        g_directed.es['distance'] = dists

    g_directed.es['weight'] = w_distribution[0](*w_distribution[1], giant.ecount())

    print("%.2f %.2f %.2f %d %d" %
          (1 - reciprocity, np.mean(giant.degree()), np.mean(g_directed.degree()),
           g_directed.vcount(), g_directed.ecount()))
    return g_directed


def get_models_to_table(n_vertices):
    models = [
        (random_partition_graph, {'n': n_vertices, 'c': 4, 'p_in': 0.35, 'p_out': 0.03}),
        (waxman, {'n': n_vertices, 'beta': 0.5, 'alpha': 0.14}),
        (random_geometric, {'n': n_vertices, 'radius': 0.13}),
        (ig.Graph.Erdos_Renyi, {'n': n_vertices, 'p': 0.1}), # mudar pra arestas ao invés de p
        (ig.Graph.Barabasi, {'n': n_vertices, 'm': 8}),
        (ig.Graph.Watts_Strogatz, {'dim': 1, 'size': 500, 'nei': 4, 'p': 0.0001}), # passar para dim 1
        (ig.Graph.Watts_Strogatz, {'dim': 1, 'size': 500, 'nei': 4, 'p': 0.002}), # nei = 3?????
        (ig.Graph.Watts_Strogatz, {'dim': 1, 'size': 500, 'nei': 4, 'p': 1}),
    ]

    alpha = 3
    plaw = lambda a, b, n: 1 - (
            (b ** (alpha + 1) - a ** (alpha + 1)) * np.random.uniform(0, 1, n) + a ** (alpha + 1)) ** (
                                   1 / (alpha + 1))

    w_dists = {'random': (np.random.uniform, (0, 1)),
               'weibull': (np.random.weibull, (5,)),
               'powerlaw': (plaw, (0.1, 1))}

    nets = dict()
    idx = 0
    for model in models:
        m_name = model[0].__name__
        print(model[0].__name__)
        for w_name, w_dist in w_dists.items():
            print(w_name)
            for _ in range(1):
                for t in [0.25, 0.5, 0.75]:
                    g = get_model_w_dir(*model, t, w_dist, w_dist)
                    giant = g.components().giant()
                    name = m_name + '_' + str(idx)
                    idx += 1
                    nets[name] = {'model': m_name + ('(p=%.4f)' % model[1]['p'] if 'Watts' in m_name else ''),
                                  'w_dist': w_name,
                                  'r': 1 - t,
                                  'w_label': 'distance' if 'distance' in giant.es.attributes() else 'weight',
                                  'simply_method1': 'mean',
                                  'simply_method2': 'mean',
                                  'net': giant
                                  }

    return nets

def to_save_gml(graph):

    str_out = 'Creator ------\ngraph\n[\ndirected 1\n'
    attr = graph.vs.attributes()
    for node in graph.vs:
        str_out += 'node\n['
        str_out += 'id ' + str(node.index) + '\n'
        for a in attr:
            str_out += a + ' ' + str(node[a]) + '\n'

        str_out += ']\n'

    attr = graph.es.attributes()
    for edge in graph.es:
        str_out += 'edge\n['
        str_out += 'source ' + str(edge.source) + '\n'
        str_out += 'target ' + str(edge.target) + '\n'
        for a in attr:
            str_out += a + ' ' + str(edge[a]) + '\n'
        str_out += ']\n'
    str_out += ']\n'

    return str_out


def get_net(args, idx):
    model, get_model, t, dist, m_name, name_dist = args
    g = get_model(*model, t, dist, None)
    giant = g.components().giant()
    return {
        'model': m_name + ('(p=%.4f)' % model[1]['p'] if 'Watts' in m_name else ''),
        'reciprocity_dist': name_dist,
        'r': 1 - t,
        'simply_method1': 'mean',
        'simply_method2': 'mean',
        'net': giant
    }


def generate_only_dir(n_vertices):
    models = [
        (random_partition_graph, {'n': n_vertices, 'c': 4, 'p_in': 0.35, 'p_out': 0.03}),
        (waxman, {'n': n_vertices, 'beta': 0.5, 'alpha': 0.14}),
        (random_geometric, {'n': n_vertices, 'radius': 0.13}),
        (ig.Graph.Erdos_Renyi, {'n': n_vertices, 'p': 0.1}),  # mudar pra arestas ao invés de p
        (ig.Graph.Barabasi, {'n': n_vertices, 'm': 8}),
        (ig.Graph.Watts_Strogatz, {'dim': 1, 'size': 500, 'nei': 4, 'p': 0.0001}),  # passar para dim 1
        (ig.Graph.Watts_Strogatz, {'dim': 1, 'size': 500, 'nei': 4, 'p': 0.002}),  # nei = 3?????
        (ig.Graph.Watts_Strogatz, {'dim': 1, 'size': 500, 'nei': 4, 'p': 1}),
    ]

    dists = {'random': (np.random.uniform, (0, 1)),
               'weibull': (np.random.weibull, (5,)),
               'powerlaw': (plaw, (0.1, 1))}

    nets = dict()
    idx = 0
    for model in models:
        m_name = model[0].__name__
        print(model[0].__name__)
        m_name += ('(p=%.4f)' % model[1]['p'] if 'Watts' in m_name else '')

        for name_dist, dist in dists.items():
            for t in np.linspace(0.01, 1, 10): # MUDAR AQUI
                net_series_key = "%s_%s_%.2f" %(m_name, name_dist, t)
                nets[net_series_key] = dict()

                pool = multiprocessing.Pool(10)
                nets_json = pool.map(partial(get_net, (model, get_model_dir, t, dist, m_name, name_dist)), np.arange(60))
                for i, net_json in enumerate(nets_json):
                    nets[net_series_key][i] = net_json
                # for i in range(3):  # MUDAR AQUI O NÚMERO DE INSTANCIAS DE CADA MODELO
                    # g = get_model_dir(*model, t, dist, None)
                    # giant = g.components().giant()
                    # nets[net_series_key][i] = {'model': m_name + ('(p=%.4f)' % model[1]['p'] if 'Watts' in m_name else ''),
                    #               'reciprocity_dist': name_dist,
                    #               'r': 1 - t,
                    #               'simply_method1': 'mean',
                    #               'simply_method2': 'mean',
                    #               'net': giant
                    #               }

    return nets


def generate_only_weight():
    pass


if __name__ == '__main__':
    # param_analysis()
    # models = get_models_dir(500, 5)  # numero de vertices, variação de r
    # save_models(models, 'dir')

    # models = get_models_to_table(500)
    # save_models(models, 'dir_w_table')

    # models = get_models_w(500, 5)
    # save_models(models, 'weighted')


    nets = generate_only_dir(500)
    save_models(nets, 'only_dir')