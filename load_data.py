import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import *
import sklearn as skl
from sklearn.model_selection import train_test_split
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler


def save_batches_to_csv():
    # columns for DataFrame
    columns_list = list(range(128)) + ['gas', 'concentration']

    # for each batch file
    for current_batch in range(1, 11):
        print(f' Saving batch: {current_batch}')

        # path to current batch file
        batch_file_path = f'data/dat_batches/batch{current_batch}.dat'

        # count amount of rows
        rows_amount = 0
        with open(batch_file_path) as f:
            for current_line in f:
                rows_amount += 1

        # create DataFrame for current batch file
        df = pd.DataFrame(index=list(range(rows_amount)), columns=columns_list)

        # read current batch file and fill df
        with open(batch_file_path) as f:
            for index, line in enumerate(f):
                lst = line.split(" ")
                gas = int(lst[0][0])
                concentration = float(lst[0].split(";")[1])
                features = [float(lst[i].split(":")[1]) for i in range(1, 129)]
                # print(features)

                for col in range(128):
                    df.loc[index, col] = features[col]
                df.loc[index, 'gas'] = gas
                df.loc[index, 'concentration'] = concentration

        # save current batch file as csv
        df = shuffle(df, random_state=RANDOM_STATE)
        df.reset_index(inplace=True, drop=True)
        #scaler = MinMaxScaler()
        #df[list(range(128))] = scaler.fit_transform(df[list(range(128))])
        df.to_csv(f'data/csv_batches/batch{current_batch}.csv')


def save_train_test_split():
    # columns for DataFrame
    # columns_list = list(range(128)) + ['gas', 'concentration']
    # samples_amount = 0
    batches_list = []

    # for each batch file
    for i in range(1, 11):
        batch_data = pd.read_csv(f'data/csv_batches/batch{i}.csv', index_col=0)
        #samples_amount += batch_data.shape[0]
        batch_data = shuffle(batch_data)
        batches_list.append(batch_data)
    data = pd.concat(batches_list)
    #print(data.head())
    #print(f' shape: {data.shape}')
    #print(data.describe())
    data = shuffle(data)

    train_data_size = 8346
    train_data = data.iloc[:train_data_size]
    test_data = data.iloc[train_data_size:]

    train_data = shuffle(train_data)
    train_data.reset_index(inplace=True, drop=True)
    test_data = shuffle(test_data)
    test_data.reset_index(inplace=True, drop=True)

    train_data.to_csv('data/train_train_data.csv')
    test_data.to_csv('data/test_test_data.csv')
    data.to_csv('data/data_data.csv')




save_train_test_split()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass