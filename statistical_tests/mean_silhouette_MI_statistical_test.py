import pandas as pd
from utils import *
import clustering_methods
from sklearn.metrics import silhouette_score
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import mutual_info_score
from scipy.stats import f_oneway
from scipy.stats import ttest_rel


def save_mean_silhouette_MI_cvs_scores(external_variable_name):
    df_mean_silhouette_MI_scores = pd.DataFrame(index=list(range(cv_amount)), columns=CLUSTERING_METHODS_NAMES_LIST)
    cvs = get_cvs(n_rows=X_test.shape[0], cv=cv_amount)

    for i, cv in enumerate(cvs):
        X_cv = X_test.iloc[cv]
        y_cv = y_test.iloc[cv]

        for clustering_method_name in CLUSTERING_METHODS_NAMES_LIST:
            print(f'{clustering_method_name}, cv: {i}')

            dims_num = clustering_methods_optimal_dims_num[clustering_method_name]
            # perform dims reduction
            ipca_test_data = X_cv
            if dims_num < FEATURES_AMOUNT:
                # perform dims reduction
                ipca = IncrementalPCA(n_components=dims_num)
                ipca_test_data = ipca.fit_transform(X_cv)

            # silhouette score
            clusters_num = clustering_methods_optimal_clusters_num[clustering_method_name]
            # perform clustering
            clustering_function = clustering_methods_functions_dict[clustering_method_name]
            clustering_method, labels = clustering_function(data=ipca_test_data,
                                                            n_clusters=clusters_num)
            sil_score = silhouette_score(X_cv, labels)


            # MI score with the best external variable
            # number of clusters should correspond to the number of classes in external variable
            clusters_num = EXTERNAL_VARIABLES_AMOUNTS_IN_TEST_DATA_DICT[external_variable_name]
            # perform clustering
            clustering_function = clustering_methods_functions_dict[clustering_method_name]
            clustering_method, pred_labels_cv = clustering_function(data=ipca_test_data,
                                                                n_clusters=clusters_num)
            true_labels_cv = y_cv[external_variable_name]
            true_labels_cv = list(true_labels_cv)
            true_labels_cv = [int(2 * true_labels_cv[i]) for i in range(len(true_labels_cv))]
            mi_score = adjusted_mutual_info_score(true_labels_cv, pred_labels_cv)

            weighted_score = 0.5 * sil_score + 0.5 * mi_score
            print(f'mi_score: {mi_score}, sil_score: {sil_score}, weighted_score: {weighted_score}')
            df_mean_silhouette_MI_scores.loc[i, clustering_method_name] = weighted_score

    df_mean_silhouette_MI_scores.to_csv('../data/mean_silhouette_MI_cvs_scores_test_data')


def save_optimal_mean_silhouette_MI_per_clustering_test_datas(external_variable_name):
    mean_silhouette_MI_optimal_per_clustering_test_data = pd.DataFrame(columns=CLUSTERING_METHODS_NAMES_LIST)
    for clustering_method_name in CLUSTERING_METHODS_NAMES_LIST:

        print(f'{clustering_method_name}, external variable: {external_variable_name}')
        dims_num = clustering_methods_optimal_dims_num[clustering_method_name]
        # perform dims reduction
        ipca_test_data = X_test
        if dims_num < FEATURES_AMOUNT:
            # perform dims reduction
            ipca = IncrementalPCA(n_components=dims_num)
            ipca_test_data = ipca.fit_transform(X_test)

        # silhouette score
        clusters_num = clustering_methods_optimal_clusters_num[clustering_method_name]
        # perform clustering
        clustering_function = clustering_methods_functions_dict[clustering_method_name]
        clustering_method, labels = clustering_function(data=ipca_test_data,
                                                        n_clusters=clusters_num)
        sil_score = silhouette_score(X_test, labels)

        # MI score with the best external variable
        # number of clusters should correspond to the number of classes in external variable
        clusters_num = EXTERNAL_VARIABLES_AMOUNTS_IN_TEST_DATA_DICT[external_variable_name]
        # perform clustering
        clustering_function = clustering_methods_functions_dict[clustering_method_name]
        clustering_method, pred_labels_cv = clustering_function(data=ipca_test_data,
                                                                n_clusters=clusters_num)
        true_labels_cv = y_test[external_variable_name]
        true_labels_cv = list(true_labels_cv)
        true_labels_cv = [int(2 * true_labels_cv[i]) for i in range(len(true_labels_cv))]
        mi_score = adjusted_mutual_info_score(true_labels_cv, pred_labels_cv)

        weighted_score = 0.5 * sil_score + 0.5 * mi_score
        print(f'mi_score: {mi_score}, sil_score: {sil_score}, weighted_score: {weighted_score}')
        mean_silhouette_MI_optimal_per_clustering_test_data.loc[0, clustering_method_name] = weighted_score

    mean_silhouette_MI_optimal_per_clustering_test_data.to_csv('../data/mean_silhouette_MI_optimal_per_clustering_test_data')


def perform_anova_test():
    scores_df = pd.read_csv('../data/mean_silhouette_MI_cvs_scores_test_data')

    # list of lists of cvs per clustering method
    scores_cvs_data = []

    for clustering_method_name in CLUSTERING_METHODS_NAMES_LIST:
        clustering_scores = scores_df[clustering_method_name].tolist()
        scores_cvs_data.append(clustering_scores)

    print(f_oneway(*scores_cvs_data))


def perform_ttest(clustering_method_name1, clustering_method_name2):
    scores_df = pd.read_csv('../data/mean_silhouette_MI_cvs_scores_test_data')
    scores1 = scores_df[clustering_method_name1]
    scores2 = scores_df[clustering_method_name2]

    print(ttest_rel(scores1, scores2, alternative='greater'))


if __name__ == '__main__':
    cv_amount = 60
    external_variable = GAS_NAME

    data = pd.read_csv('../data/test_test_data.csv', index_col=0)
    X_test = data.drop(EXTERNAL_VARIABLES_NAMES, axis=1)
    y_test = data[EXTERNAL_VARIABLES_NAMES]

    clustering_methods_optimal_dims_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_DIMS_NUM
    clustering_methods_optimal_clusters_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_CLUSTERS_NUM
    clustering_methods_functions_dict = clustering_methods.CLUSTERING_METHODS_FUNCTIONS_DICT

    # save_mean_silhouette_MI_cvs_scores(external_variable_name=external_variable)
    #save_optimal_mean_silhouette_MI_per_clustering_test_datas(external_variable_name=external_variable)
    #perform_anova_test()
    perform_ttest('kmeans', 'minibatchkmeans')