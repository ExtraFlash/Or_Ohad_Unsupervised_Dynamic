import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
import clustering_methods
from sklearn.metrics import silhouette_score
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter

from matplotlib import style

matplotlib.rc('font', family='Times New Roman')


def plot_bar(data, x_label, y_label, title, bottom, upperBound: float = 1):
    #style.use('ggplot')

    #plt.figure(figsize=(8, 11))
    barWidth = 0.6

    left = np.array(range(4))
    height = []
    for clustering_method_name in CLUSTERING_METHODS_NAMES_LIST:
        score = data.loc[0, clustering_method_name]
        height.append(score - bottom)

    #ax.ylim([bottom, 1])
    plt.ylim([bottom, upperBound])

    tick_label = []
    for clustering_method_name in CLUSTERING_METHODS_NAMES_LIST:
        tick_label.append(CLUSTERING_METHODS_PLOT_NAMES_DICT[clustering_method_name])

    plt.bar(left, height, color=CLUSTERING_COLORS, width=barWidth, bottom=bottom, tick_label=tick_label)

    if x_label:
        plt.xlabel(x_label, fontsize=13)
    # naming the y-axis
    plt.ylabel(y_label, fontsize=13)
    # plot title
    plt.title(title, fontsize=13)
    #ax.savefig(f'../figures/mean_silhouette_MI_optimal_per_cluster', pad_inches=0.2, bbox_inches="tight")
    #ax.show()


def plot_bar_MI_external_variables():
    MI_gas = pd.read_csv('../data/MI_gas_optimal_per_clustering_test_data')
    MI_concenctrations = pd.read_csv('../data/MI_concentration_optimal_per_clustering_test_data')

    gmm_scores = []
    gmm_scores.append(MI_gas.loc[0, 'gmm'])
    gmm_scores.append(MI_concenctrations.loc[0, 'gmm'])

    kmeans_scores = []
    kmeans_scores.append(MI_gas.loc[0, 'kmeans'])
    kmeans_scores.append(MI_concenctrations.loc[0, 'kmeans'])

    birch_scores = []
    birch_scores.append(MI_gas.loc[0, 'birch'])
    birch_scores.append(MI_concenctrations.loc[0, 'birch'])

    minibatchkmeans = []
    minibatchkmeans.append(MI_gas.loc[0, 'minibatchkmeans'])
    minibatchkmeans.append(MI_concenctrations.loc[0, 'minibatchkmeans'])

    ind = np.arange(len(gmm_scores))  # the x locations for the groups
    width = 0.1  # the width of the bars

    plt.bar(ind - width * 3 / 2, gmm_scores, width,
            label=CLUSTERING_METHODS_PLOT_NAMES_DICT['gmm'],
            color=['royalblue'],
            edgecolor = 'black')
    plt.bar(ind - width / 2, kmeans_scores, width,
            label=CLUSTERING_METHODS_PLOT_NAMES_DICT['kmeans'],
            color=['slategrey'],
            edgecolor = 'black')
    plt.bar(ind + width / 2, birch_scores, width,
            label=CLUSTERING_METHODS_PLOT_NAMES_DICT['birch'],
            color=['limegreen'],
            edgecolor = 'black')
    plt.bar(ind + width * 3 / 2, minibatchkmeans, width,
            label=CLUSTERING_METHODS_PLOT_NAMES_DICT['minibatchkmeans'],
            color=['deeppink'],
            edgecolor = 'black')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('MI', fontsize=13)
    plt.ylim([0., 1.])
    plt.title('B', fontsize=13)
    plt.xticks(ind, ['Gas', 'Concentration'], fontsize=13)
    # plt.xlabel('G1', 'G2', 'G3', 'G4', 'G5')
    plt.legend()


def silhouette_per_point_plot():

    n_clusters = clustering_methods_optimal_clusters_num['kmeans']
    n_dims = clustering_methods_optimal_dims_num['kmeans']

    test_data = pd.read_csv('../data/test_test_data.csv')
    X_test = test_data.drop(EXTERNAL_VARIABLES_NAMES, axis=1)

    ipca = IncrementalPCA(n_components=n_dims)
    ipca_test_data = ipca.fit_transform(X_test)

    kmeans, y_pred = clustering_methods.kmeans(data=ipca_test_data,
                                               n_clusters=n_clusters)
    silhouette_coefficients = silhouette_samples(ipca_test_data, y_pred)
    silhouette_score_ = silhouette_score(X_test, y_pred)

    padding = len(ipca_test_data) // 30
    pos = padding
    ticks = []
    for i in range(n_clusters):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()

        color = matplotlib.cm.Spectral(i / n_clusters)

        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                      facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(n_clusters)))
    plt.ylabel("Cluster")

    plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xlabel("Silhouette Coefficient", fontsize=13)

    plt.axvline(x=silhouette_score_, color="red", linestyle="--")
    #plt.title("$Number of clusters={}$".format(n_clusters), fontsize=16)
    plt.title("D", fontsize=13)






if __name__ == '__main__':
    clustering_methods_optimal_dims_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_DIMS_NUM
    clustering_methods_optimal_clusters_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM


    fig = plt.figure(figsize=(13, 10))
    #a1 = plt.subplot2grid((2, 2), (0, 0))
    #a2 = plt.subplot2grid((2, 2), (0, 1))
    #a3 = plt.subplot2grid((2, 2), (1, 0))
    #a4 = plt.subplot2grid((2, 2), (1, 1))

    # plot silhouette score for every clustering method
    plt.subplot(221)
    plot_data = pd.read_csv('../data/silhouette_optimal_per_clustering_test_data', index_col=0)
    plot_bar(plot_data, x_label="", y_label='Silhouette score',
             title='A', bottom=-1)

    # plot MI with clustering methods and genres
    plt.subplot(222)
    # path = f'../data/MI_{GAS_NAME}_optimal_per_clustering_test_data'
    # plot_data = pd.read_csv(path, index_col=0)
    # plot_bar(plot_data, x_label="", y_label='MI score',
    #          title='MI between clustering methods and Genders', bottom=0., upperBound=1.)
    plot_bar_MI_external_variables()

    # plot weighted score with clustering methods and genres
    plt.subplot(223)
    plot_data = pd.read_csv('../data/mean_silhouette_MI_optimal_per_clustering_test_data', index_col=0)
    plot_bar(plot_data, x_label="", y_label='Weighted Silhouette and MI',
             title='C', bottom=0)
    fig.tight_layout(pad=2)

    plt.subplot(224)
    silhouette_per_point_plot()
    plt.savefig('../figures/Fig1', pad_inches=0.2, bbox_inches="tight")
    plt.show()