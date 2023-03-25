import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from utils import *
from sklearn.utils import shuffle
from sklearn.metrics import silhouette_score

#df = pd.read_csv('data/csv_batches/batch1.csv')
#a = int(0.36 * 7)
#print(a)
#print(df['concentration'].unique())

#df2 = pd.read_csv('first10.csv')
#print(df2['concentration'].unique())

#df1 = pd.DataFrame(index=[0,1,2,3], columns=['a','b','c'])
#df1.loc[0,'a'] = 3
#df1.loc[2,'b'] = 5

#X = df1.drop(['b','c'], axis=1)
#print(X)
#df2 = pd.DataFrame(index=[0,1,2,3], columns=['a','b','c'])
#df2.iloc[[0,1,2,3]] = df1.copy()
#df2.loc[0, 'a'] = 5
#print(df1)
#concentrations = set()
#for i in range(1, 11):
#    df = pd.read_csv(f'data/csv_batches/batch{i}.csv')
#    current = set(df[CONCENTRATION_NAME].unique())
#    concentrations.update(current)

#print(concentrations)

# check if silhouette method gets affected by the index in dataframe
df = pd.DataFrame(index=[0,1,2, 3, 4, 5, 6, 7], columns=['a','b','c'])
df.iloc[0] = [1,2,3]
df.iloc[1] = [4,5,6]
df.iloc[2] = [7,8,9]
df.iloc[3] = [7,6,5]
df.iloc[4] = [14,13,12]
df.iloc[5] = [1.1,8.8,7.6]
df.iloc[6] = [12,23,7]
df.iloc[7] = [15,17,20]

#print(df)

# scaler = StandardScaler()
# df[['a', 'b']] = scaler.fit_transform(df[['a', 'b']])

#print(df)

#from sklearn.metrics.cluster import adjusted_mutual_info_score
#from sklearn.metrics.cluster import mutual_info_score
#v1 = [0, 0, 7, 5 ,1,2,3,4,5]
##v1 = np.array(v1)
#v2 = [0, 0, 1, 1,2,2,3,4,2.5]
#v2 = [int(2*v2[i]) for i in range(len(v2))]
#v1_int = [int(2 * v1[i]) for i in range(4)]
#v2_int = [int(2 * v2[i]) for i in range(4)]
#print(adjusted_mutual_info_score(v1, v2))
# x = ['a']*20
# y = [2] * 19 + [3]
# print(x)
# print(y)
# print(adjusted_mutual_info_score(x, y))


#concentrations = set()
#df = pd.read_csv(f'data/test_test_data.csv')
#print(f' amount of concentrations: {len(set(df[CONCENTRATION_NAME].unique()))}')
#df = shuffle(df)
#df = shuffle(df)
#df = shuffle(df)
#df = shuffle(df)
#df = shuffle(df)
#df = shuffle(df)
#df = shuffle(df)

#df.reset_index(inplace=True, drop=True)
#df.to_csv('data/shuffled_test_data')

ind = np.arange(10)
print(ind)