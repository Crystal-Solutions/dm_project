from __future__ import print_function
from __future__ import division

from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib
#%matplotlib inline

import pandas as pd
import numpy as np



from sklearn.model_selection import train_test_split
import statsmodels.api as sm



# just for the sake of this blog post!
from warnings import filterwarnings
filterwarnings('ignore')

def shift(df,n):
    df = df.shift(n)
    df.fillna(method='bfill', inplace=True)
    return df

# load the provided data
train_features = pd.read_csv('data-processed/dengue_features_train.csv',
                             index_col=[0,1,2])

train_labels = pd.read_csv('data-processed/dengue_labels_train.csv',
                           index_col=[0,1,2])

train_features.drop('week_start_date', axis=1, inplace=True)
#print(train_features.columns.values)

for i in train_features.columns.values:
    name = i + "_shifted"
    train_features[name] = shift(train_features[i], 8)

train_features.drop('total_cases_shifted', axis=1, inplace=True)
#print(train_features.columns.values)

#print(train_features.head(5))

# Seperate data for San Juan
sj_train_features = train_features.loc['sj']
sj_train_labels = train_labels.loc['sj']

# Separate data for Iquitos
iq_train_features = train_features.loc['iq']
iq_train_labels = train_labels.loc['iq']

'''print('San Juan')
print('features: ', sj_train_features.shape)
print('labels  : ', sj_train_labels.shape)

print('\nIquitos')
print('features: ', iq_train_features.shape)
print('labels  : ', iq_train_labels.shape)'''

#print(sj_train_features.head(5))

# Remove `week_start_date` string.
#sj_train_features.drop('week_start_date', axis=1, inplace=True)
#iq_train_features.drop('week_start_date', axis=1, inplace=True)

# Null check
#print(pd.isnull(sj_train_features).any())

#Fill missing values
sj_train_features.fillna(method='ffill', inplace=True)
iq_train_features.fillna(method='ffill', inplace=True)

#print(pd.isnull(sj_train_features).any())

sj_train_features['total_cases'] = sj_train_labels.total_cases
iq_train_features['total_cases'] = iq_train_labels.total_cases

sj_correlations = sj_train_features.corr()
iq_correlations = iq_train_features.corr()

for i in sj_correlations.columns.values:
    if not i.endswith('shifted'):
        if i.endswith('cases'): continue
        else:
            print(i)
            sj_correlations.drop(i, axis=1, inplace=True)

sj_corr_heat = sns.heatmap(sj_correlations)
plt.title('San Juan Variable Correlations')
plt.show()

iq_corr_heat = sns.heatmap(iq_correlations)
plt.title('IQ Variable Correlations')
#plt.show()

sj_correlations = sj_train_features.corr()
iq_correlations = iq_train_features.corr()

# San Juan
(sj_correlations
     .total_cases
     .drop('total_cases') # don't compare with myself
     .sort_values(ascending=False)
     .plot
     .barh())

# Iquitos
'''(iq_correlations
     .total_cases
     .drop('total_cases') # don't compare with myself
     .sort_values(ascending=False)
     .plot
     .barh())'''
