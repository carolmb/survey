import metrics
import itertools
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import igraph as ig
from scipy.stats import pearsonr
from scipy import stats
from functools import partial
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from generate_models import load_models, get_bases_infos, get_bases_graphs
from generate_models import get_models_to_table

from scipy.stats import variation

legends = ['n_dir_n_weig', 'n_dir_y_weig', 'y_dir_n_weig', 'y_dir_y_weig']
n_metrics = 4


def norm(X):
    X = np.asarray(X)
    X = np.nan_to_num(X)
    Y = (X - np.nanmean(X)) / np.nanstd(X)
    return Y


def plot_lda_by_graph(Z, labels, graphs):
    begin = 0
    print('z', Z.shape)

    for name, family in graphs.items():
        N = 0
        # print(begin)

        for graph in family:
            N += graph['net'].vcount()

        # print(N)
        if N == 0:
            continue

        lda = LinearDiscriminantAnalysis(n_components=2)
        lda.fit(Z[begin:begin + N * n_metrics], labels[begin:begin + N * n_metrics])
        X_trans = lda.transform(Z[begin:begin + N * n_metrics])

        scatter_dist = -1
        # scatter_dist = metrics.Score(Z[begin:begin + n_metrics * vcount],
        #                              np.asarray([vcount] * n_metrics))

        lda1 = X_trans[:, 0]
        lda2 = X_trans[:, 1]

        dataframe = pd.DataFrame({'lda1': lda1, 'lda2': lda2, 'graph': labels[begin:begin + N * n_metrics]})

        fig = sns.jointplot(data=dataframe, x='lda1', y='lda2', hue='graph',
                            palette=sns.color_palette("tab20"),
                            s=5)  # cm.get_cmap('tab20b', n_metrics*len(family)), s=5)

        fig.ax_joint.legend(bbox_to_anchor=(0.65, 1), loc=2, prop={'size': 6})
        plt.suptitle(name + '\n(scatter distance = %.2f)' % scatter_dist)

        print(name, lda.scalings_[:, 0])
        fig.ax_joint.set_xlabel('LDA1 %.2f' % (lda.explained_variance_ratio_[0] * 100))
        fig.ax_joint.set_ylabel('LDA2 %.2f' % (lda.explained_variance_ratio_[1] * 100))

        plt.savefig('imgs/pca_lda/' + name + '_' + type_net + '_lda_by_graph.pdf')
        plt.show()
        begin += N * n_metrics


def plot_pca_by_graph(Z, labels, type_net):
    begin = 0
    for name, graph in graphs.items():
        if graph['w_label'] != type_net:
            continue
        vcount = graph['net'].vcount()
        j = 0

        pca = PCA(n_components=2)
        pca.fit(Z[begin:begin + n_metrics * vcount], labels[begin:begin + n_metrics * vcount])
        X_trans = pca.transform(Z[begin:begin + n_metrics * vcount])

        scatter_dist = metrics.Score(Z[begin:begin + N * vcount],
                                     np.asarray([vcount] * N))

        pca1 = X_trans[:, 0]
        pca2 = X_trans[:, 1]

        graph_class = [legends[i] for i in range(n_metrics) for _ in range(vcount)]
        dataframe = pd.DataFrame({'pca1': pca1, 'pca2': pca2, 'graph': graph_class})
        fig = sns.jointplot(data=dataframe, x='pca1', y='pca2', hue='graph',
                            palette=sns.color_palette("pastel")[:n_metrics], s=5)

        plt.suptitle(name + '\n(scatter distance = %.2f)' % scatter_dist)
        fig.ax_joint.set_xlabel('PCA1 %.2f' % (pca.explained_variance_ratio_[0] * 100))
        fig.ax_joint.set_ylabel('PCA2 %.2f' % (pca.explained_variance_ratio_[1] * 100))
        # plt.savefig('imgs/' + name + '_pca_by_graph_log.pdf')
        # plt.show()
        begin += n_metrics * vcount


def plt_hists(graphs, Y, metric_name, type_net):
    begin = 0

    for name, family in graphs.items():
        for graph_data in family:
            if graph_data['w_label'] != type_net:
                continue
            graph = graph_data['net']
            vcount = graph.vcount()
            Y_temp = Y[begin:begin + vcount * 4]
            Y_temp = np.asarray(Y_temp).reshape(-1, vcount)
            fig, axs = plt.subplots(1, 4, figsize=(16, 4.5), tight_layout=True, sharex=True, sharey=True)
            i = 0
            for group, label in zip(Y_temp, legends):
                axs[i].hist(group)
                axs[i].set_title(label)
                i += 1

            fig.suptitle(name + ' ' + metric_name)
            plt.savefig('imgs/hists_syn_data/' + name + '_' + metric_name + '.pdf')
            # plt.show()

            begin += vcount * n_metrics


def sperman_corr_graphs(Y, vcount):
    y_temp = np.asarray(Y).reshape(-1, vcount)
    for i, j in itertools.combinations(np.arange(len(y_temp)), 2):
        corr = stats.spearmanr(y_temp[i], y_temp[j])
        print('spearman', i, j, corr)


def sperman_corr(graphs, Y, metrics):
    names = [m.__name__ for m in metrics]
    begin = 0

    for name, graph in graphs.items():
        print(name)
        vcount = graph.vcount()
        Y_temp = np.asarray(Y[begin:begin + vcount * n_metrics])
        j = 0
        for i in range(0, len(Y_temp), vcount):
            print(legends[j])
            j += 1
            m1 = Y_temp[i:i + vcount, 0]
            m2 = Y_temp[i:i + vcount, 1]
            m3 = Y_temp[i:i + vcount, 2]
            m4 = Y_temp[i:i + vcount, 3]

            corr = stats.spearmanr(m1, m2)
            print('%s %s %.2f' % (names[0], names[1], corr[0]))
            corr = stats.spearmanr(m1, m3)
            print('%s %s %.2f' % (names[0], names[2], corr[0]))
            corr = stats.spearmanr(m2, m3)
            print('%s %s %.2f' % (names[1], names[2], corr[0]))

            corr = stats.spearmanr(m1, m4)
            print('%s %s %.2f' % (names[0], names[3], corr[0]))
            corr = stats.spearmanr(m2, m4)
            print('%s %s %.2f' % (names[1], names[3], corr[0]))
            corr = stats.spearmanr(m3, m4)
            print('%s %s %.2f' % (names[2], names[3], corr[0]))

        begin += vcount * n_metrics


'''
m = [
    g1,
    g2, 
    g3
    ] linhas
    
'''


def get_tree(graphs):
    tree = dict()
    tree['similarity'] = []

    for key, graph in graphs.items():
        if graph['net'] != '':
            tree['similarity'].append(graph)

    return tree


def _plot_metric_by_reciprocity(nets, metric_name, xlabel, suffix):
    metric_by_net = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
    for net in nets:
        metric_by_net[net['family']]['ndnw'][net['r']] = net['ndnw']
        metric_by_net[net['family']]['yw'][net['r']] = net['yw']
        # metric_by_net[net['family']]['y_dir_n_w'][net['r']] = net['y_dir_n_w']
        # metric_by_net[net['family']]['y_dir_y_w'][net['r']] = net['y_dir_y_w']

    for family, family_net in metric_by_net.items():
        for net_name, r_values in family_net.items():
            print(net_name)
            x = []
            y = []
            ye = []
            for r, values in r_values.items():
                values = np.asarray(values)
                x.append(r)
                v = np.ma.masked_equal(np.nan_to_num(values), 0)
                if set(values) != {0}:
                    y.append(np.mean(v))
                else:
                    y.append(0)
                ye.append(np.std(np.nan_to_num(values)) / 5)

            idxs = np.argsort(x)
            x = np.asarray(x)
            y = np.asarray(y)
            x = x[idxs]
            y = y[idxs]

            plt.errorbar(x, y, yerr=ye, label=family + ' ' + net_name)  # net_name # + ' ' + family)
    plt.title(net_name)
    plt.legend(fontsize='small')
    plt.xlabel(xlabel)
    plt.ylabel(metric_name)
    plt.tight_layout()
    plt.savefig('imgs/reciprocity/%s_giant_hmean_%s%s.pdf' % (metric_name, net_name, suffix))
    plt.show()


def plot_metric_by_reciprocity(graphs_tree, metrics_by_type, xlabel, suffix=''):
    for type_net, nets in graphs_tree.items():

        labels = []
        for data in nets:
            graph = data['net']
            for m in range(n_metrics):
                labels += [legends[m] + ' ' + data['name'] + ' ' + data['family']] * graph.vcount()

        for i, metric in enumerate(metrics_by_type[type_net]):
            for data in nets:
                X = metric(data, 'yw')  # undir_unweig, weig, in_degree, in_strength
                data['ndnw'] = list(metric(data, 'nwnd'))
                data['yw'] = list(X)
                # data['y_dir_n_w'] = X[2]
                # data['y_dir_y_w'] = X[3]

            _plot_metric_by_reciprocity(nets, metric.__name__, xlabel, suffix)


def plot_joint(graphs_tree):
    metrics_distance = [metrics.get_degree, metrics.get_betweenness,
                        metrics.get_closeness]
    metrics_similarities = [metrics.get_degree, metrics.get_evcent,
                            metrics.get_pagerank, metrics.get_constraint]  # , metrics.get_katz]
    metrics_by_type = {'dist': metrics_distance, 'weight': metrics_similarities}

    for type_net, nets in graphs_tree.items():
        N = 0
        for net in nets:
            N += net['net'].vcount()

        Z = np.zeros((N * n_metrics, len(metrics_by_type[type_net])))

        labels = []
        for data in nets:
            graph = data['net']
            for m in range(n_metrics):
                labels += [legends[m] + ' ' + data['name'] + ' ' + data['family']] * graph.vcount()

        for i, metric in enumerate(metrics_by_type[type_net]):
            Y = []
            for data in nets:
                X = metric(data)  # undir_unweig, weig, in_degree, in_strength
                X = flatten(X)
                Y += X
                # sperman_corr_graphs(X, graph['net'].vcount())

            # normalizar entre as redes para cada medida
            # plt_hists(graphs, Y, metric.__name__, type_net)

            Y = norm(np.asarray(Y))
            # Y = norm(np.log(1 + np.asarray(Y)))

            Z[:, i] = Y

        plot_lda_by_graph(Z, labels, nets)

        # np.savetxt('medidas.csv', Z, delimiter=',')

        # https://igraph-help.nongnu.narkive.com/38qXaASY/warning-message-in-eigen-centrality-function-of-package-igraph


def _plot_arrows(Z, labels, label_names, title, suffix=''):
    labels = np.asarray(labels)
    pca = PCA(n_components=2, svd_solver='arpack', random_state=3)
    pca.fit(Z)
    print(pca.components_)
    print(pca.explained_variance_)

    Z_transf = pca.transform(Z)
    x_transf = Z_transf[:, 0]
    y_transf = Z_transf[:, 1]

    idx_color = sorted(list(set(label_names.values())))

    print(idx_color)
    kcolors = 6
    colors = []
    for name in ['Blues', 'Greens', 'Reds', 'Greys', 'BuPu']:
        color_map = plt.get_cmap(name)
        colors += [color_map(i) for i in np.linspace(0.3, 0.9, kcolors)]

    # colors = plt.cm.tab20.colors + plt.cm.Dark2.colors + plt.cm.Set3.colors

    plt.figure(figsize=(16, 8))
    for begin in np.arange(0, max(labels), 11):
        X = []
        Y = []
        print(label_names[begin + 1])
        for l in np.arange(begin, begin + 11):
            x = x_transf[labels == l]
            y = y_transf[labels == l]
            # plt.scatter(x, y, alpha=0.02, c='gray', s=1, marker='o')
            x = np.mean(x)
            y = np.mean(y)
            # print(x, y)
            X.append(x)
            Y.append(y)
        # print()
        plt.plot(X[2:], Y[2:], marker='x', c=colors[idx_color.index(label_names[begin + 1])],
                 alpha=0.7, lw=1.2, ls='dashed')  # label=label_names[begin])
        line = plt.arrow(X[2], Y[2], X[1] - X[2], Y[1] - Y[2], alpha=0.7, lw=1.2, ls='dashed',
                         color=colors[idx_color.index(label_names[begin + 1])], head_width=0.02)

        plt.scatter(X[:1], Y[:1], c=colors[idx_color.index(label_names[begin])], marker='+', s=25, alpha=0.7)

    #
    handles = [mlines.Line2D([], [], color=colors[idx], label=label) for idx, label in enumerate(idx_color)]
    plt.legend(handles, idx_color, prop={'size': 8}, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)

    plt.xlabel('PCA1 %.2f%%' % (pca.explained_variance_ratio_[0] * 100))
    plt.ylabel('PCA2 %.2f%%' % (pca.explained_variance_ratio_[1] * 100))
    # plt.xlim((-4, 4))
    # plt.ylim((-3, 3))
    plt.title(title + ' models')
    plt.tight_layout()
    # plt.savefig('arrows_%s_v3%s.pdf' % (title, suffix))
    plt.show()


def plot_arrows(graphs_tree, metrics_by_type, graph_prop, suffix=''):
    for type_net, nets in graphs_tree.items():
        N = 0
        N_nwnd = 0
        for net in nets:
            N += net['net'].vcount()
            if net['r'] == 1:
                N_nwnd += net['net'].vcount()

        Z = np.zeros((N + N_nwnd, len(metrics_by_type[type_net])))
        for i, metric in enumerate(metrics_by_type[type_net]):
            Y = []
            j = 0
            labels = []
            map_names = dict()
            print(metric.__name__)
            for data in nets:
                if data['r'] == 1:
                    X = metric(data, 'nwnd')  # undir_unweig, weig, in_degree, in_strength
                    Y += list(X)
                    labels += [j] * len(X)
                    map_names[j] = data['family'] + ' (nwnd)'
                    j += 1
                X = metric(data, graph_prop)  # undir_unweig, weig, in_degree, in_strength
                Y += list(X)  # com direção e sem peso
                labels += [j] * len(X)
                map_names[j] = data['family']
                j += 1
            Y = norm(Y)
            print(min(Y))
            Y = np.log(4 + Y)
            Z[:, i] = Y

        _plot_arrows(Z, labels, map_names, type_net, suffix)


def graph_props(graph_data, label):
    g = graph_data['net']
    graph_simply = metrics.dir2undir(g, graph_data['w_label']).simplify(multiple=True, loops=False,
                                                                        combine_edges=graph_data['simply_method2'])
    graph_simply = graph_simply.components().giant()
    if label == 'nwnd':
        return graph_simply.vcount(), graph_simply.ecount()
    elif label == 'yw':
        return graph_simply.vcount(), graph_simply.ecount()
    elif label == 'yd':
        return g.vcount(), g.ecount()
    elif label == 'ywyd':
        return g.vcount(), g.ecount()


def graph_metrics(graph_data, label):
    g = graph_data['net']
    graph_simply = metrics.dir2undir(g, graph_data['w_label']).simplify(multiple=True, loops=False,
                                                                        combine_edges=graph_data['simply_method2'])
    graph_simply = graph_simply.components().giant()

    if label == 'nwnd':
        return graph_simply.assortativity_degree(directed=False)
    elif label == 'yw':
        return graph_simply.assortativity_degree(directed=False)
    elif label == 'yd':
        return g.assortativity_degree(directed=True)
    elif label == 'ywyd':
        return g.assortativity_degree(directed=True)


def pairs_cv(values):
    line1 = values[0]
    val_sum = 0
    count = 0
    for val in variation(values, axis=0):
        val_sum += val
        count += 1
    val_sum /= count

    pairs = []
    for val in line1:
        pairs.append((val, val_sum))

    return pairs


def is_saved(net_name):
    import glob
    files = glob.glob('metrics_saved/*.csv')
    for file in files:
        if net_name in file:
            return True
    return False


def get_values_saved(net_name):
    return np.loadtxt('metrics_saved/%s.csv' % net_name)


def calculate_save_metrics(metrics, arg):
    net_name, data = arg
    output = open('metrics_saved/%s.csv' % net_name, 'w')
    for graph_prop in ['nwnd', 'yd']:  # 'yw', 'ywyd']:
        for name, (metric, _) in metrics.items():
            for i, instance_data in data.items():
                X = metric(instance_data, graph_prop)
                output.write('%s\t%s\t%s\n' % (name, graph_prop, ','.join(["%f" % x for x in X])))


def get_resume(metrics, row):
    metric = row['metric']
    val = [float(v) for v in row['values'].split(',')]
    resume_func = metrics[metric][1]
    return resume_func(val)


import multiprocessing

def plot_table(nets, metrics, outtable, typedata):
    metrics_names = list(metrics.keys())
    table = []
    cvs = []
    cvs_rows = []
    pool = multiprocessing.Pool(10)
    pool.map(partial(calculate_save_metrics, metrics), list(nets.items()))
    # for net_name, data in nets.items():
    #     print(net_name)
    #     # if not is_saved(net_name):
    #     #     print('aqui')
    #     calculate_save_metrics(net_name, data, metrics)

'''
        X = pd.read_csv('metrics_saved/%s.csv' % net_name, sep='\t', header=None)
        saved_metrics = X[0].values[:len(metrics_names)]
        if tuple(saved_metrics) != tuple(metrics_names):
            calculate_save_metrics(net_name, data, metrics)
            X = pd.read_csv('metrics_saved/%s.csv' % net_name, sep='\t', header=None)

        X.columns = ['metric', 'graph_type', 'values']
        X['resume'] = X.apply(partial(get_resume, metrics), axis=1)
        rows = []
        for grapy_type, graph_metrics in X.groupby('graph_type'):
            resume = graph_metrics['resume'].values.reshape(30, -1).mean(axis=0)
            print(resume.shape)
            rows.append([net_name.replace('_', ' '), grapy_type] + resume.tolist())#graph_metrics['resume'].values.tolist())

        rows.append(np.arange(len(metrics_names)).tolist())

        table_net = pd.DataFrame(rows, index=np.arange(0, 3))
        table_net.columns = ['network', 'type'] + metrics_names

        cv = table_net[table_net.columns[2:]][:-1].apply(variation, axis=0)
        cvs.append(cv.values.tolist())
        table_net.iloc[-1] = ['', 'cv'] + cv.values.tolist()
        table_net['<cv>'] = [np.nan, np.nan, np.mean(cv)]
        cvs_rows.append(np.mean(cv))
        table.append(table_net)

        # fazer correlação entre degree e constraint

    ids = np.arange(len(cvs_rows))  # np.argsort(cvs_rows)
    original = [t['network'].iloc[0] for t in table]

    table = [table[i] for i in ids]

    cvs_rows = [cvs_rows[i] for i in ids]
    net_names = [t['network'].iloc[0] for t in table]

    fig, ax = plt.subplots(figsize=(10, 5))

    if typedata == 'real':
        color_map = plt.get_cmap('hsv')
        colors = [color_map(d) for d in np.linspace(0.01, 0.9, len(net_names))]
        d = 0
        for r, l in zip(cvs_rows, net_names):
            if '0.50' in l:
                continue
            ax.scatter([r], [1], alpha=0.6, label=l, color=colors[d])
            d += 1
    else:
        colors = []
        kcolors = 10
        for name in ['Blues', 'Greens', 'Reds',
                     'Greys', 'BuPu', 'winter', 'autumn', 'BrBG']:
            color_map = plt.get_cmap(name)
            colors += [color_map(i) for i in np.linspace(0.3, 0.9, kcolors)]
        colormap = dict()
        original = [o for o in original if '0.50' not in o]
        for i, key in enumerate(original):
            print(key)
            colormap[key] = colors[i]

        f, axs = plt.subplots(7, 1, sharex=True, figsize=(8, 10))
        for i, rval in enumerate(['W Strogatz(p=0.0001)', 'W Strogatz(p=0.0020)',
                                  'W Strogatz(p=1.0000)', 'Er', 'Barabasi', 'random partition', 'waxman']):
            icolor = 0
            for r, l in zip(cvs_rows, net_names):
                if '0.5' in l:
                    continue
                if rval in l:
                    axs[i].scatter([r], [1], alpha=0.6, label=l, color=colormap[l])
                    axs[i].get_yaxis().set_visible(False)
                    icolor += 1
            axs[i].legend(fontsize='xx-small', bbox_to_anchor=(1.1, 0.8), loc='right',
                          ncol=2)
        f.tight_layout()
        f.savefig('csv_%s_byfamily.pdf' % typedata)

    # fazer para as medidas tbm, dados syn e real

    table = pd.concat(table)

    cvs_mean = np.mean(cvs, axis=0)

    print(cvs_mean)
    ids = np.argsort(cvs_mean)

    lastline = dict()
    for name, val in zip(metrics_names, cvs_mean):
        lastline[name] = val

    fig, ax = plt.subplots(figsize=(10, 5))

    color_map = plt.get_cmap('hsv')
    colors = [color_map(d) for d in np.linspace(0.01, 0.9, len(metrics_names))]
    d = 0
    for r, l in zip(cvs_mean, metrics_names):
        ax.scatter([r], [1], alpha=0.6, label=l, color=colors[d])
        d += 1
    plt.legend(ncol=2)
    ax.get_yaxis().set_visible(False)
    plt.savefig('csv_metrics_%s.pdf' % typedata)

    lastline['network'] = ''
    lastline['type'] = '<cv>'
    lastline['<cv>'] = np.nan

    lastline_pd = pd.DataFrame(lastline, index=[0])

    table = table.append(lastline_pd)

    table[table.columns[2:-1]] = table[table.columns[2 + ids]]
    table.columns = ['network', 'type'] + table.columns[2 + ids].tolist() + ['cv']

    table.to_latex(open(outtable, 'w'), float_format="%.2f", index_names=False, na_rep='', index=False)

    print(table.head(10))

    # separar os modelos por r ou familia, ajustar o grau medio para ficar parecido, tirar coluna ecount

    # for i in range(number_of_metrics):
    #     points = []
    #     for pair in pairs:
    #         points.append(pair[i])
    #     X = np.asarray([p[0] for p in points])
    #     Y = np.asarray([p[1] for p in points])
    #     plt.scatter(X, Y)
    #     plt.xlabel('std')
    #     plt.ylabel('<cv>')
    #     print(X, Y)
    #     X = X[~np.isnan(Y)]
    #     Y = Y[~np.isnan(Y)]
    #     plt.title("%s (pearson=%.2f)" % (metric_names[i], pearsonr(X, Y)[0]))
    #     plt.show()
'''

import glob



import io
from xnet_temp import from_xnet_to_igraph


def get_syn_data(folder):
    jsonfile = glob.glob('data/models/%s/models.json' % folder)[0]

    with open(jsonfile, 'r') as inputfile:
        model_infos = json.load(inputfile)
    for key, infos in model_infos.items():
        str_gml = open('data/models/%s.xnet' % key, 'r').read().split('-------')[:-1]

        idx = 0
        print(key, len(str_gml))
        for i, str_gml_instance in enumerate(str_gml):
            handler = io.StringIO(str_gml_instance)
            net = from_xnet_to_igraph(handler)
            infos[str(idx)]['net'] = net
            infos[str(idx)]['w_label'] = 'weight'
            idx += 1

    return model_infos

def get_arg_0(x):
    return x[0]


def get_arg_1(x):
    return x[1]


if __name__ == '__main__':
    # valid_nets = get_bases_infos()
    # print(valid_nets)
    # graphs = get_bases_graphs()
    # print(len(graphs))

    flatten = lambda t: [item for sublist in t for item in sublist]

    metrics_distance = [metrics.degree, metrics.betweenness,
                        metrics.closeness]  # metrics.shorç.test_path]

    metrics_similarities = {'vcount': (graph_props, get_arg_0),
                            'ecount': (graph_props, get_arg_1),
                            'degree': (metrics.degree, np.std),
                            'evcent': (metrics.evcent, np.std),
                            'pagerank': (metrics.pagerank, np.std),  # max
                            'constraint': (metrics.constraint, np.std),
                            'clustering coef': (metrics.clustering_coef, np.std),
                            'betweenness': (metrics.betweenness, np.std),
                            'closeness': (metrics.closeness, np.std),
                            'triangle': (metrics.triangle, np.std),
                            # 'small world': (metrics.small_world_propensity, lambda x: x),
                            # 'katz': (metrics.get_katz, np.std)
                            }

    # plot_metric_by_reciprocity(graphs_tree, metrics_by_type, 'threshold','_weights')

    # graphs = load_models('dir')
    # graphs_tree = get_tree(graphs)
    # plot_arrows(graphs_tree, metrics_by_type, 'yd', '_dir')

    # graphs = load_models('weighted')
    # graphs_tree = get_tree(graphs)
    # plot_arrows(graphs_tree, metrics_by_type, 'yw', '_weights_v2')

    # graphs_tree = get_bases_graphs()
    # plot_table(graphs_tree, metrics_similarities, 'temp/tablametrics_realnets.tex', 'real')

    graphs_tree = get_syn_data('only_dir')
    plot_table(graphs_tree, metrics_similarities, 'temp/tablametrics_synnets.tex', 'syn')

'''
curtose e tal <- pagerank
eigenv tbm precisa de outra medida de localização


reunião 13-09: olhar outras medidas de redes TODO
e verificar a correlação das medidas com o grau DONE

'''
