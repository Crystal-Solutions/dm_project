# -*- coding: utf-8 -*-
"""
Created on Wed May 24 20:12:30 2017

@author: Janaka
"""


import pandas as pd
DATA_PATH = './data/original_data/'
CITY_NAMES = ['sj','iq']
BM_FEATURE_NAMES =  ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c']
LABEL_COLUMN_NAME = 'total_cases'

def getFeatureNames():
    df = pd.read_csv(DATA_PATH+'dengue_features_train.csv', index_col=[0, 1, 2])
    lst = list(df.columns.values)
    lst.remove('week_start_date')
    return lst
def getData(feature_list=False):
#    Load training data'
    df = pd.read_csv(DATA_PATH+'dengue_features_train.csv', index_col=[0, 1, 2])
    df.fillna(method='ffill', inplace=True)
    
    df_test = pd.read_csv(DATA_PATH+'dengue_features_test.csv', index_col=[0, 1, 2])
    df_test.fillna(method='ffill', inplace=True)
    
    
    if(feature_list):
        df = df[feature_list]
        df_test = df_test[feature_list]
    
        
    labels = pd.read_csv(DATA_PATH+'dengue_labels_train.csv', index_col=[0, 1, 2])

    
    submission = pd.read_csv(DATA_PATH+"submission_format.csv",
                     index_col=[0, 1, 2])
    
    return df,labels,df_test,submission
    

