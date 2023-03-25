import pandas as pd
import numpy as np
import clustering_methods
from utils import *
from sklearn.metrics import silhouette_score


def find_num_of_clusters_silhouette(clustering_method_name):
    max_score = 0
    best_clusters_num = 0
    df = pd.DataFrame(columns=CLUSTERS_NUM_LIST)
    for clusters_num in CLUSTERS_NUM_LIST:
        print(f'{clustering_method_name}, clusters amount: {clusters_num}')

        scores = []
        for batch in range(1, BATCHES_AMOUNT + 1):
            train_data_batch = pd.read_csv(f'data/csv_batches/train_data_batch{batch}.csv', index_col=0)
            X_train_batch = train_data_batch.drop(EXTERNAL_VARIABLES_NAMES, axis=1)

            clustering_function = clustering_methods_functions_dict[clustering_method_name]
            clustering_method, labels = clustering_function(X_train_batch, n_clusters=clusters_num)
            score = silhouette_score(X_train_batch, labels)
            scores.append(score)

        mean_score = np.mean(scores)
        if mean_score > max_score:
            max_score = mean_score
            best_clusters_num = clusters_num
        df.loc[0, clusters_num] = mean_score
    df.to_csv(f'data/silhouette_best_clusters_num_train_data/{clustering_method_name}')
    print(f'Best amount of clusters for {clustering_method_name} is: {best_clusters_num}, score: {max_score}')


if __name__ == '__main__':
    #data = pd.read_csv('data/train_data.csv')
    #X = data.drop(EXTERNAL_VARIABLES_NAMES, axis=1)
    #print(X.head())

    #clusters_num_list = list(range(2, CLUSTERS_NUM_LIST + 1))  # from 2 to the max number of clusters

    clustering_methods_functions_dict = clustering_methods.CLUSTERING_METHODS_FUNCTIONS_DICT

    # find optimal number of clusters for all clustering methods
    for clustering_method_name_ in CLUSTERING_METHODS_NAMES_LIST:
        find_num_of_clusters_silhouette(clustering_method_name_)