import numpy as np
import pandas as pd
from utils import *
import clustering_methods
from sklearn.metrics import silhouette_score
from sklearn.decomposition import IncrementalPCA


def find_num_of_dims_silhouette(clustering_method_name):
    max_score = 0
    best_dims_num = 0
    df = pd.DataFrame(columns=DIMS_NUM_LIST)

    for dims_num in DIMS_NUM_LIST:
        print(f'{clustering_method_name}, dims_num = {dims_num}')

        scores = []
        for batch in range(1, BATCHES_AMOUNT + 1):
            train_data_batch = pd.read_csv(f'data/csv_batches/train_data_batch{batch}.csv', index_col=0)
            X_train_batch = train_data_batch.drop(EXTERNAL_VARIABLES_NAMES, axis=1)

            ipca_train_data = X_train_batch
            # if number of dimensions is smaller than number of features
            if dims_num < FEATURES_AMOUNT:
                # perform dims reduction
                ipca = IncrementalPCA(n_components=dims_num)
                ipca_train_data = ipca.fit_transform(X_train_batch)

            # perform clustering
            clustering_function = clustering_methods_functions_dict[clustering_method_name]
            clustering_method, labels = clustering_function(data=ipca_train_data,
                                                            n_clusters=clustering_methods_optimal_clusters_num[
                                                                clustering_method_name])
            score = silhouette_score(X_train_batch, labels)
            scores.append(score)

        mean_score = np.mean(scores)
        if mean_score > max_score:
            max_score = mean_score
            best_dims_num = dims_num
        df.loc[0, dims_num] = mean_score
    df.to_csv(f'data/silhouette_best_dims_num_train_data/{clustering_method_name}')
    print(f'Best amount of dims for {clustering_method_name} is: {best_dims_num}, score: {max_score}')


if __name__ == '__main__':
    # data = pd.read_csv('data/train_data.csv')
    # X = data.drop(EXTERNAL_VARIABLES_NAMES, axis=1)
    # y = data[[utils.GENRE_TOP_NAME]]

    clustering_methods_optimal_clusters_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM
    clustering_methods_functions_dict = clustering_methods.CLUSTERING_METHODS_FUNCTIONS_DICT

    # find optimal number of dims for all clustering methods
    for clustering_method_name_ in CLUSTERING_METHODS_NAMES_LIST:
        find_num_of_dims_silhouette(clustering_method_name_)
