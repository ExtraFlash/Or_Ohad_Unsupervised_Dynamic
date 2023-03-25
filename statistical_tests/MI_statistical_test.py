import numpy as np
import pandas as pd
from utils import *
import clustering_methods
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import adjusted_mutual_info_score
from scipy.stats import alexandergovern
from scipy.stats import kruskal
from scipy.stats import f_oneway
from scipy.stats import ttest_rel
from scipy.stats import mannwhitneyu


def perform_anova_test(external_variable_name):
    mi_scores_per_clustering_list = []
    for clustering_methods_name in CLUSTERING_METHODS_NAMES_LIST:
        df = pd.read_csv(f'../data/MI/{external_variable_name}/clustering_methods/{clustering_methods_name}')
        mi_scores = df[external_variable_name].tolist()
        print(f' clustering: {clustering_methods_name}, external: {external_variable_name}, scores: {mi_scores}')
        mi_scores_per_clustering_list.append(mi_scores)

    print(f_oneway(*mi_scores_per_clustering_list))


# saves MI scores for each clustering method with the gender, in the test data
def save_scores(external_variable_name):
    # MI_optimal_per_clustering_test_data
    df_mi_optimal_per_clustering_test = pd.DataFrame(columns=CLUSTERING_METHODS_NAMES_LIST)

    data = pd.read_csv('../data/test_test_data.csv', index_col=0)
    X_test = data.drop(external_variable_name, axis=1)
    y_test = data[external_variable_name]

    for clustering_method_name in CLUSTERING_METHODS_NAMES_LIST:
        print(f'Starting: {clustering_method_name}')
        dims_num = clustering_methods_optimal_dims_num[clustering_method_name]
        clusters_num = EXTERNAL_VARIABLES_AMOUNTS_IN_TEST_DATA_DICT[external_variable_name]

        # perform dims reduction
        ipca = IncrementalPCA(n_components=dims_num)
        ipca_test_data = ipca.fit_transform(X_test)

        # perform clustering
        clustering_function = clustering_methods_functions_dict[clustering_method_name]
        clustering_method, pred_labels_cv = clustering_function(data=ipca_test_data,
                                                                n_clusters=clusters_num)
        mi_score = adjusted_mutual_info_score(y_test, pred_labels_cv)
        df_mi_optimal_per_clustering_test.loc[0, clustering_method_name] = mi_score
    df_mi_optimal_per_clustering_test.to_csv(f'../data/MI_{external_variable_name}_optimal_per_clustering_test_data')


# between the best clusterings for the same external variable
def perform_ttest(clustering_method_name1, clustering_method_name2, external_variable_name):
    df_scores1 = pd.read_csv(f'../data/MI/{external_variable_name}/clustering_methods/{clustering_method_name1}')
    df_scores2 = pd.read_csv(f'../data/MI/{external_variable_name}/clustering_methods/{clustering_method_name2}')

    scores1 = df_scores1[external_variable_name].tolist()
    scores2 = df_scores2[external_variable_name].tolist()

    print(mannwhitneyu(scores1, scores2, alternative='greater'))


# between the best clusterings, each from different external variable
# first for gas, second for concentration
def perform_ttest_different_external_variable(clustering_method_name1, clustering_method_name2):
    df_scores1 = pd.read_csv(f'../data/MI/{GAS_NAME}/clustering_methods/{clustering_method_name1}')
    df_scores2 = pd.read_csv(f'../data/MI/{CONCENTRATION_NAME}/clustering_methods/{clustering_method_name2}')

    scores1 = df_scores1[GAS_NAME].tolist()
    scores2 = df_scores2[CONCENTRATION_NAME].tolist()

    print(mannwhitneyu(scores1, scores2, alternative='greater'))


if __name__ == '__main__':
    clustering_methods_optimal_dims_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_DIMS_NUM
    clustering_methods_optimal_clusters_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM
    clustering_methods_functions_dict = clustering_methods.CLUSTERING_METHODS_FUNCTIONS_DICT

    #perform_anova_test(CONCENTRATION_NAME)


    # perform anova test for every external variable independently
    # and find the best clustering method for each using t test
    # then another t test between the best two, each from different external variable

    # perform_anova_test()
    #save_scores(GAS_NAME)
    #save_scores(CONCENTRATION_NAME)
    #perform_ttest('gmm', 'minibatchkmeans', external_variable_name=CONCENTRATION_NAME)

    perform_ttest_different_external_variable('gmm', 'birch')

