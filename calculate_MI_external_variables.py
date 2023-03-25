import pandas as pd
from utils import *
import clustering_methods
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
import numpy as np


def calc_MI(clustering_method_name, external_variable_name, cv_amount=10):
    np.random.seed(RANDOM_STATE)
    df = pd.DataFrame(index=list(range(cv_amount)), columns=[external_variable_name])
    cvs = get_cvs(n_rows=X_test.shape[0], cv=cv_amount)
    dims_num = clustering_methods_optimal_dims_num[clustering_method_name]
    save_dir = f'data/MI/{external_variable_name}/clustering_methods'
    # set clusters amount as amount of classes in external variable
    clusters_num = EXTERNAL_VARIABLES_AMOUNTS_IN_TEST_DATA_DICT[external_variable_name]

    for i, cv in enumerate(cvs):
        X_cv = X_test.iloc[cv]
        y_cv = y_test.iloc[cv]
        print(f'{clustering_method_name}, cv: {i}')

        # perform dims reduction
        ipca = IncrementalPCA(n_components=dims_num)
        ipca_test_data = ipca.fit_transform(X_cv)

        # perform clustering
        clustering_function = clustering_methods_functions_dict[clustering_method_name]
        clustering_method, pred_labels_cv = clustering_function(data=ipca_test_data,
                                                                n_clusters=clusters_num)
        true_labels_cv = y_cv[external_variable_name]

        # true_labels_cv contains 2.5 a float number
        # we will multiply the labels by 2 and cast to int
        # it won't affect the mutual information score
        true_labels_cv = list(true_labels_cv)
        true_labels_cv = [int(2 * true_labels_cv[i]) for i in range(len(true_labels_cv))]
        #print(f'external_variable: {external_variable_name}, clustering: {clustering_method_name}')
        #print(true_labels_cv)
        #print(pred_labels_cv)

        mi_score = mutual_info_score(true_labels_cv, pred_labels_cv)
        df.loc[i, external_variable_name] = mi_score

    df.to_csv(f'{save_dir}/{clustering_method_name}')


def calc_MI_test(external_variable_name, cv_amount):
    np.random.seed(RANDOM_STATE)
    cvs = get_cvs(n_rows=X_test.shape[0], cv=cv_amount)
    clusters_num = EXTERNAL_VARIABLES_AMOUNTS_IN_TEST_DATA_DICT[external_variable_name]

    for clustering_method_name in CLUSTERING_METHODS_NAMES_LIST:
        df = pd.DataFrame(index=list(range(cv_amount)), columns=[external_variable_name])
        for i, cv in enumerate(cvs):
            X_cv = X_test.iloc[cv]
            y_cv = y_test.iloc[cv]

            print(f'{clustering_method_name}, cv: {i}')

            dims_num = clustering_methods_optimal_dims_num[clustering_method_name]
            # perform dims reduction
            ipca_test_data = X_cv
            if dims_num < FEATURES_AMOUNT:
                # perform dims reduction
                ipca = IncrementalPCA(n_components=dims_num)
                ipca_test_data = ipca.fit_transform(X_cv)


            # MI score with the best external variable
            # perform clustering
            clustering_function = clustering_methods_functions_dict[clustering_method_name]
            clustering_method, pred_labels_cv = clustering_function(data=ipca_test_data,
                                                                    n_clusters=clusters_num)
            true_labels_cv = y_cv[external_variable_name]
            true_labels_cv = list(true_labels_cv)
            true_labels_cv = [int(2 * true_labels_cv[i]) for i in range(len(true_labels_cv))]
            mi_score = adjusted_mutual_info_score(true_labels_cv, pred_labels_cv)

            #print(f'mi_score: {mi_score}, sil_score: {sil_score}, weighted_score: {weighted_score}')
            df.loc[i, external_variable_name] = mi_score

        df.to_csv(f'data/MI/{external_variable_name}/clustering_methods/{clustering_method_name}')


if __name__ == '__main__':
    data = pd.read_csv('data/test_test_data.csv', index_col=0)
    X_test = data.drop(EXTERNAL_VARIABLES_NAMES, axis=1)
    y_test = data[EXTERNAL_VARIABLES_NAMES]

    clustering_methods_optimal_dims_num = clustering_methods.CLUSTERING_METHODS_OPTIMAL_DIMS_NUM
    clustering_methods_functions_dict = clustering_methods.CLUSTERING_METHODS_FUNCTIONS_DICT

    cvs_amounts_dict = {GAS_NAME: 9,
                        CONCENTRATION_NAME: 50}

    #for external_variable in EXTERNAL_VARIABLES_NAMES:
    #    for clustering_method_name_ in CLUSTERING_METHODS_NAMES_LIST:
     #       cv_amount = cvs_amounts_dict[external_variable]
      #      calc_MI(clustering_method_name_, cv_amount=cv_amount, external_variable_name=external_variable)
    #calc_MI_test(CONCENTRATION_NAME, cv_amount=60)
    calc_MI_test(GAS_NAME, cv_amount=9)