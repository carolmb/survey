import metrics
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from scipy.ndimage import gaussian_filter

from scipy.stats import pearsonr
from scipy import stats
from functools import partial
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from generate_models import load_models, get_bases_infos, get_bases_graphs
from generate_models import get_models_to_table

from main import get_resume
from functools import partial
from scipy.stats import variation


def graph_props(graph_data, label):
    g = graph_data['net']
    return g.vcount(), g.ecount()


metrics_similarities = {'vcount': (graph_props, lambda x: x[0]),
                        'ecount': (graph_props, lambda x: x[1]),
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

def get_syn_data():
    models = load_models('only_dir')
    models2 = dict()
    for name, infos in models.items():
        custom_name = infos['family'] + 'r=%.2f ' % (infos['r']) + infos['reciprocity_dist']
        if 'graph' in custom_name:
            custom_name = custom_name.replace('graph', '')
        models2[custom_name] = infos

    return models2


def is_saved(net_name):
    import glob
    files = glob.glob('metrics_saved/*.csv')
    for file in files:
        if net_name in file:
            return True
    return False


def calculate_save_metrics(net_name, data, metrics, prop):
    output = open('metrics_saved/%s.csv' % net_name, 'w')
    for net in data:
        for name, (metric, _) in metrics.items():
            X = metric(net, prop)
            output.write('%s\t%.2f\t%s\n' % (name, net['r'], ','.join(["%f" % x for x in X])))


def generate_metrics(prop):
    metrics_names = list(metrics_similarities.keys())

    models = get_syn_data()

    for r_dist in reciprocity_dist:
        for model_name in models_names:
            model_family = []
            for net_name, infos in models.items():
                if infos['reciprocity_dist'] == r_dist and infos['family'] == model_name:
                    infos['w_label'] = None
                    model_family.append(infos)

            family = model_name +'_' + r_dist + '_' + prop
            print(family)
            if not is_saved(family):
                calculate_save_metrics(family, model_family, metrics_similarities, prop)

            X = pd.read_csv('metrics_saved/%s.csv' % family, sep='\t', header=None)
            saved_metrics = X[0].values[:len(metrics_names)]
            if tuple(saved_metrics) != tuple(metrics_names):
                calculate_save_metrics(family, model_family, metrics_similarities, prop)
                X = pd.read_csv('metrics_saved/%s.csv' % family, sep='\t', header=None)


def save_tables(prop):
    for model_name in models_names:
        for r_dist in reciprocity_dist:
            family = model_name + '_' + r_dist + '_' + prop
            x = pd.read_csv('metrics_saved/%s.csv' % family, sep='\t', header=None)
            x.columns = ['metric', 'reciprocity', 'values']
            table = []
            R = []
            for name, group in x.groupby('reciprocity'):
                R.append(name)
                resume_values = defaultdict(lambda: [])
                for _, row in group.iterrows():
                    values = [float(v) for v in row['values'].split(",")]
                    resume_val = metrics_similarities[row['metric']][1](values)
                    resume_values[row['metric']].append(resume_val)

                row = pd.DataFrame(dict(resume_values))
                table.append(row)

            table = pd.concat(table)
            cvs = table.apply(variation)
            ave_cvs = variation(cvs)
            table['reciprocity'] = R

            table = table.append(cvs, ignore_index=True)
            table.iloc[-1, -1] = ave_cvs

            table.to_latex(family + '.tex', header=True, na_rep='', float_format="%.2f",
                                 index_names=False, index=False, caption=model_name)


reciprocity_dist = {'weibull', 'powerlaw', 'random'}
models_names = {'random_partition_graph', 'Erdos_Renyi', 'Watts_Strogatz(p=1.0000)',
                'waxman', 'Barabasi', 'Watts_Strogatz(p=0.0001)',
                'random_geometric', 'Watts_Strogatz(p=0.0020)'}

reci_values = ['0.01', '0.12', '0.23', '0.34', '0.45', '0.56', '0.67', '0.78', '0.89', '1.00']

def gaussian_curve(prop):
    metrics_similarities = {'vcount': (graph_props, lambda x: x[0]),
                            'ecount': (graph_props, lambda x: x[1]),
                            'degree': (metrics.degree, np.std),
                            'evcent': (metrics.evcent, np.std),
                            'pagerank': (metrics.pagerank, np.std),  # max
                            'constraint': (metrics.constraint, np.std),
                            'clustering coef': (metrics.clustering_coef, np.std),
                            'betweenness': (metrics.betweenness, np.std),
                            'closeness': (metrics.closeness, np.std),
                            'triangle': (metrics.triangle, np.std),
                            }
    density = defaultdict(lambda: [])

    for r_dist in reciprocity_dist:
        for model_name in models_names:
            reci_means = defaultdict(lambda: [])
            for val in reci_values:
                file = model_name + '_' + r_dist + '_' + val
                x = pd.read_csv('metrics_saved/%s.csv' % file, sep='\t', header=None)
                x.columns = ['metric', 'prop', 'values']
                x = x[x['prop'] == prop]
                for metric, metric_values in x.groupby('metric'):
                    m = metric_values.apply(partial(get_resume, metrics_similarities), axis=1)
                    reci_means[metric].append(m.values)

            for metric, values in reci_means.items():
                cv = variation(values)
                density[metric].append(cv)

    for name, hist in density.items():
        y, b = np.histogram(hist, bins=np.linspace(0, 1, 50))
        y = gaussian_filter(y, sigma=1)
        plt.plot((b[:-1] + b[1:])/2, y, label=name, alpha=0.5)

    plt.legend()
    plt.savefig('cv_hist_%s.pdf' % prop)
    plt.clf()


if __name__ == '__main__':
    gaussian_curve('yd')
    gaussian_curve('nwnd')