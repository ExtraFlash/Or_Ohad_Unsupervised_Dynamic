from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import BayesianGaussianMixture

##### CONSTANTS
CLUSTERING_METHODS_NAMES_LIST = ['gmm', 'kmeans', 'birch', 'minibatchkmeans']
# CLUSTERING_METHODS_NAMES_LIST = ['kmeans', 'birch', 'minibatchkmeans']

CLUSTERING_METHODS_PLOT_NAMES_DICT = {'gmm': 'GMM',
                                      'kmeans': 'KMeans',
                                      'birch': 'Birch',
                                      'minibatchkmeans': 'Minibatch KMeans'}

CLUSTERING_METHODS_DICT = {'gmm': GaussianMixture(),
                           'kmeans': KMeans(),
                           'birch': Birch(),
                           'minibatchkmeans': MiniBatchKMeans()}

CLUSTERING_COLORS = ['royalblue', 'slategrey', 'limegreen', 'deeppink']

ANOMALY_DETECTION_MODELS = ['isolation_forest']

DIMENSIONALITY_REDUCTIONS_NAMES_LIST = ['ipca']
DATA_SIZE = 13910
TRAIN_DATA_PERCENTAGE = 0.6
TEST_DATA_PERCENTAGE = 0.4
RANDOM_STATE = 42
GASES_AMOUNT = 6
CONCENTRATIONS_MAX_AMOUNT = 59
BATCHES_AMOUNT = 10

CLUSTERS_NUM_LIST = lst = list(range(5, 60, 3))  # from 5 to 59, step size of 3
DIMS_NUM_LIST = [5, 10, 20, 50, 70, 90, 128]
FEATURES_AMOUNT = 128

##### LABEL NAMES
EXTERNAL_VARIABLES_NAMES = ['gas', 'concentration']
EXTERNAL_VARIABLES_AMOUNTS_DICT = {'gas': 6,
                                   'concentration': 59}
EXTERNAL_VARIABLES_AMOUNTS_IN_TEST_DATA_DICT = {'gas': 6,
                                                'concentration': 57}
GAS_NAME = 'gas'
CONCENTRATION_NAME = 'concentration'


##### FUNCTIONS


def get_cvs(n_rows, cv=10):
    cvs = []
    increment = n_rows // cv
    current_end = increment
    for i in range(cv):
        index_list = list(range(current_end - increment, current_end))
        cvs.append(index_list)
        current_end += increment
    return cvs
