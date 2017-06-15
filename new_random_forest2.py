# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 13:09:57 2017

@author: Janaka
"""

import itertools
import pandas as pd
import numpy as np
import dengue_data as dd
import dengue_processing as dp
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

ef_lst = dd.BM_FEATURE_NAMES +['reanalysis_min_air_temp_k','reanalysis_relative_humidity_percent','reanalysis_precip_amt_kg_per_m2'] #All features
label_col = dd.LABEL_COLUMN_NAME
cityNames = dd.CITY_NAMES
train,labels, test, submission = dd.getData(ef_lst)


features  = train.loc['sj'],train.loc['iq']
labels = labels.loc['sj'],labels.loc['iq']
test_features = test.loc['sj'],test.loc['iq']


def shift(df):
    df['reanalysis_relative_humidity_percent_2'] = dp.shift(df['reanalysis_relative_humidity_percent'],8)
    df['reanalysis_relative_humidity_percent_3'] = dp.shift(df['reanalysis_relative_humidity_percent'],3)
    df['reanalysis_precip_amt_kg_per_m2_2'] = dp.shift(df['reanalysis_precip_amt_kg_per_m2'],8)
    df['reanalysis_precip_amt_kg_per_m2_3'] = dp.shift(df['reanalysis_precip_amt_kg_per_m2'],6)
    df['reanalysis_specific_humidity_g_per_kg_2'] = dp.shift(df['reanalysis_specific_humidity_g_per_kg'],2)
    df['reanalysis_specific_humidity_g_per_kg_3'] = dp.shift(df['reanalysis_specific_humidity_g_per_kg'],6)#11
    df['reanalysis_dew_point_temp_k_2'] = dp.shift(df['reanalysis_dew_point_temp_k'],11)#2,9
    df['reanalysis_dew_point_temp_k_3'] = dp.shift(df['reanalysis_dew_point_temp_k'],5)
    df['reanalysis_dew_point_temp_k_4'] = dp.shift(df['reanalysis_dew_point_temp_k'],6)

#Shift 
sj_features = features[0]
sj_test = test_features[0]

shift(sj_features)
shift(sj_test)


predictions = []
for i,(f,l,t) in enumerate(zip(features,labels,test_features)):

#    #Pick best matching
#    f['total_cases'] = l
#    for i in range(1,5):
#        for feature in ef_lst:
#            f[feature+'_'+str(i)] =  dp.shift(f.loc[:,(feature)],i)
#            t[feature+'_'+str(i)] =  dp.shift(t.loc[:,(feature)],i)
#    cor = f.corr().total_cases.drop('total_cases').sort_values(ascending=False)
#    selected_features = []
#    for i,v in cor.items():
#        if(v>0.2):
#            selected_features.append(i)
#    f = f.loc[:,selected_features]
##    t = t.loc[:,selected_features]
                
    rgr = RandomForestRegressor(n_estimators=100)
#    rgr = DecisionTreeRegressor(max_depth=4)
#    rgr = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,5), random_state=1)

    if (i==1):
        f = f.loc[:,dd.BM_FEATURE_NAMES]
        t = t.loc[:,dd.BM_FEATURE_NAMES]

    #Data for a Sample Try
    size = int(f.shape[0]*8/10)
    f_t = f.head(size)
    f_t_test = f.tail(f.shape[0] - size)
    l_t = l.head(size)
    l_t_test = l.tail(l.shape[0] - size)
    
    pca = PCA(n_components=4)   
    pca.fit(np.concatenate([f_t]))
    f_t = pca.transform(f_t)
    f_t_test = pca.transform(f_t_test)
    f = pca.transform(f)
    t = pca.transform(t)
    #Try
    rgr.fit(f_t,np.ravel(l_t))
    p = list(map(int,map(round,rgr.predict(f_t_test))))
    er = mean_absolute_error(l_t_test,p)
    print(er)
    
    rgr.fit(f,np.ravel(l));
    prd = list(map(int,map(round,rgr.predict(t))))
    predictions.append(prd)
#    break
        
#Sace data
submission.total_cases = np.concatenate(predictions)
submission.to_csv(dd.DATA_PATH+"new_random_forest_bin.csv")