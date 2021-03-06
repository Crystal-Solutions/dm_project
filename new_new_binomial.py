# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:39:11 2017

@author: Shanika Ediriweera
"""

from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

import dengue_data as dd
import dengue_processing as dp
from sklearn.metrics import mean_absolute_error

# just for the sake of this blog post!
from warnings import filterwarnings
filterwarnings('ignore')

# load the provided data
train_features = pd.read_csv('./data/dengue_features_train.csv', index_col=[0,1,2])

train_labels = pd.read_csv('./data/dengue_labels_train.csv', index_col=[0,1,2])

# Seperate data for San Juan
sj_train_features = train_features.loc['sj']
sj_train_labels = train_labels.loc['sj']

# Separate data for Iquitos
iq_train_features = train_features.loc['iq']
iq_train_labels = train_labels.loc['iq']


# Remove `week_start_date` string.
sj_train_features.drop('week_start_date', axis=1, inplace=True)
iq_train_features.drop('week_start_date', axis=1, inplace=True)


sj_train_features.fillna(method='ffill', inplace=True)
iq_train_features.fillna(method='ffill', inplace=True)


sj_train_features['total_cases'] = sj_train_labels.total_cases
iq_train_features['total_cases'] = iq_train_labels.total_cases
   
##########################              
# compute the correlations
sj_correlations = sj_train_features.corr()
iq_correlations = iq_train_features.corr()

sj_correlations = sj_train_features.corr(method='spearman')
iq_correlations = iq_train_features.corr(method='spearman')

# plot san juan heat matrix
#sj_corr_heat = sns.heatmap(sj_correlations)
#plt.title('San Juan Variable Correlations')

# San Juan
#(sj_correlations
#     .total_cases
#     .drop('total_cases') # don't compare with myself
#     .sort_values(ascending=False)
#     .plot
#     .barh())

# Iquitos
#(iq_correlations
#     .total_cases
#     .drop('total_cases') # don't compare with myself
#     .sort_values(ascending=False)
#     .plot
#     .barh())
##########################
df= None
df_sj=None
df_iq=None
def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    
#    # fill missing values
#    df.fillna(method='ffill', inplace=True)
#
#    # add labels to dataframe
#    if labels_path:
#        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
#        df = df.join(labels)

#    'reanalysis_air_temp_k',
#    'station_max_temp_c',
#    reanalysis_sat_precip_amt_mm
#    precipitation_amt_mm

    #ADD SHIFTED FEATURES HERE
    df['reanalysis_relative_humidity_percent_2'] = dp.shift(df['reanalysis_relative_humidity_percent'],8)
    df['reanalysis_relative_humidity_percent_3'] = dp.shift(df['reanalysis_relative_humidity_percent'],3)
    df['reanalysis_precip_amt_kg_per_m2_2'] = dp.shift(df['reanalysis_precip_amt_kg_per_m2'],8)
    df['reanalysis_precip_amt_kg_per_m2_3'] = dp.shift(df['reanalysis_precip_amt_kg_per_m2'],6)
    df['reanalysis_specific_humidity_g_per_kg_2'] = dp.shift(df['reanalysis_specific_humidity_g_per_kg'],2)
    df['reanalysis_specific_humidity_g_per_kg_3'] = dp.shift(df['reanalysis_specific_humidity_g_per_kg'],6)#11
    df['reanalysis_dew_point_temp_k_2'] = dp.shift(df['reanalysis_dew_point_temp_k'],11)#2,9
    df['reanalysis_dew_point_temp_k_3'] = dp.shift(df['reanalysis_dew_point_temp_k'],5)
    df['reanalysis_dew_point_temp_k_4'] = dp.shift(df['reanalysis_dew_point_temp_k'],6)
    
    #CHANGE HERE ---- SJ FEATURES
    features_sj = ['reanalysis_specific_humidity_g_per_kg', 
             'reanalysis_dew_point_temp_k', 
             'station_avg_temp_c', 
             'reanalysis_air_temp_k',
             'station_max_temp_c',
             'reanalysis_relative_humidity_percent',
             'reanalysis_relative_humidity_percent_2',
             'reanalysis_relative_humidity_percent_3',
             'reanalysis_precip_amt_kg_per_m2_2',
             'reanalysis_precip_amt_kg_per_m2_3',
             'reanalysis_specific_humidity_g_per_kg_2',
             'reanalysis_specific_humidity_g_per_kg_3',
             'reanalysis_dew_point_temp_k_2',
             'reanalysis_dew_point_temp_k_3',
             'reanalysis_dew_point_temp_k_4']
    
    #CHANGE HERE ---- IQ FEATURES
    features_iq = ['reanalysis_specific_humidity_g_per_kg', 
             'reanalysis_dew_point_temp_k', 
             'station_avg_temp_c', 
             'station_min_temp_c',
             'reanalysis_min_air_temp_k']
    
    
    
    df_sj = df[features_sj]
    df_iq = df[features_iq]
    
    # fill missing values
    df_sj.fillna(method='ffill', inplace=True)
    df_iq.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df_sj = df_sj.join(labels)
        df_iq = df_iq.join(labels)
    
    # separate san juan and iquitos
    sj = df_sj.loc['sj']
    iq = df_iq.loc['iq']
    
    return sj, iq

sj_train, iq_train = preprocess_data('./data/part_removed/dengue_features_train.csv',
                                    labels_path="./data/part_removed/dengue_labels_train_filled_94_anom.csv")
#print(sj_train.describe())

sj_train_subtrain = sj_train.head(800)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)

iq_train_subtrain = iq_train.head(400)
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)

from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA

def get_best_model_sj(train, test):
    # Step 1: specify the form of the model
                               
#
#    #CHANGE HERE ---- SJ FEATURES   
#    letters = 'ABCDEFGHIJKLMNOP'
#    model_formula = "total_cases ~ 1  " 
#    for i in range(15):
#        model_formula += '+ '+letters[i]
                    
                 
#    #CHANGE HERE ---- SJ FEATURES   
    model_formula = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "station_avg_temp_c + " \
                    "station_max_temp_c + " \
                    "reanalysis_air_temp_k + " \
                    "reanalysis_relative_humidity_percent + " \
                    "reanalysis_relative_humidity_percent_2 + " \
                    "reanalysis_relative_humidity_percent_3 + " \
                    "reanalysis_precip_amt_kg_per_m2_2 + " \
                    "reanalysis_precip_amt_kg_per_m2_3 + " \
                    "reanalysis_specific_humidity_g_per_kg_2 + " \
                    "reanalysis_specific_humidity_g_per_kg_3 + " \
                    "reanalysis_dew_point_temp_k_2 + " \
                    "reanalysis_dew_point_temp_k_3 + " \
                    "reanalysis_dew_point_temp_k_4"
                    
                    
                    
    grid = 10 ** np.arange(-8, -3, dtype=np.float64)
                    
    best_alpha = []
    best_score = 1000
        
    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)
            
    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model

def get_best_model_iq(train, test):
    # Step 1: specify the form of the model

    
    #CHANGE HERE ---- IQ FEATURES
    model_formula = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "station_min_temp_c + " \
                    "station_avg_temp_c + " \
                    "reanalysis_min_air_temp_k"
                    
                                      
    grid = 10 ** np.arange(-8, -3, dtype=np.float64)
                    
    best_alpha = []
    best_score = 1000
        
    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)
            
    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model


##PCA
#pca = PCA(n_components=15)   
#PCA_COLS = list('ABCDEFGHIJKLMNOP'[:15])
#sj_train_subtrain_for_pca = sj_train_subtrain.copy()
#sj_train_subtrain_for_pca = sj_train_subtrain_for_pca.drop('total_cases', 1)
#pca.fit(sj_train_subtrain_for_pca)
#
##principal component analysis
#sj_train_subtrain = pd.concat([pd.DataFrame(pca.transform(sj_train_subtrain_for_pca),columns=PCA_COLS,index=sj_train_subtrain_for_pca.index),sj_train_subtrain],axis=1)
#sj_train_subtest = pd.concat([pd.DataFrame(pca.transform(sj_train_subtest.drop('total_cases', 1)),columns=PCA_COLS,index=sj_train_subtest.index),sj_train_subtest],axis=1)

#sj_train_subtrain = sj_train_subtrain.loc[:1993]
#idx = pd.IndexSlice
#sj_train_subtrain = pd.concat([sj_train_subtrain.loc[:1993],sj_train_subtrain.loc[1995:]])


sj_best_model = get_best_model_sj(sj_train_subtrain, sj_train_subtest)
iq_best_model = get_best_model_iq(iq_train_subtrain, iq_train_subtest)

sj_test, iq_test = preprocess_data('./data/dengue_features_test.csv')
#sj_test = pd.concat([pd.DataFrame(pca.transform(sj_test),columns=PCA_COLS,index=sj_test.index),sj_test],axis=1)

sj_predictions = sj_best_model.predict(sj_test).astype(int)
iq_predictions = iq_best_model.predict(iq_test).astype(int)

submission = pd.read_csv("./data/submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("./data/janaka_1.csv")