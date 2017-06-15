# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 21:03:48 2017

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
from sklearn.neural_network import MLPRegressor

ef_lst = dd.BM_FEATURE_NAMES +['reanalysis_min_air_temp_k'] #All features
label_col = dd.LABEL_COLUMN_NAME
cityNames = dd.CITY_NAMES
train,labels, test, submission = dd.getData(ef_lst)


features  = train.loc['sj'],train.loc['iq']
labels = labels.loc['sj'],labels.loc['iq']
test_features = test.loc['sj'],test.loc['iq']

#Shift 
sj_featurs = features[0]
sj_test = test_features[0]



##Code to play with
#results = []
#for shifts in itertools.product(range(5),repeat=5):
#    f,l,t  = (features[0].copy(),labels[0],test_features[0])
#    for i,shift_value in enumerate(shifts):
#        f.loc[:,(ef_lst[i])] = dp.shift(f.loc[:,(ef_lst[i])],shift_value)
#        
#    rgr = RandomForestRegressor(min_samples_split = 250,n_estimators=140)
##    rgr = MLPRegressor(hidden_layer_sizes=(100, ))
#    
#    #Data for a Sample Try
#    size = int(f.shape[0]*8/10)
#    f_t = f.head(size)
#    f_t_test = f.tail(f.shape[0] - size)
#    l_t = l.head(size)
#    l_t_test = l.tail(l.shape[0] - size)
#    
#    #Try
#    rgr.fit(f_t,np.ravel(l_t))
#    prd = list(map(int,map(round,rgr.predict(f_t_test))))
#    er = mean_absolute_error(l_t_test,prd)
#    print(er,shifts)
#    results.append((er,shifts))
#             
#
#print(min(results,key=lambda x:x[0]))



    
##Code to run and get result to submit
shifts = [2,2,4,2,2]
for i,(f,v) in enumerate(zip(ef_lst,shifts)):
    sj_featurs.loc[:,(f+str(i))] = dp.shift(sj_featurs.loc[:,(f)],v)
    sj_test.loc[:,(f+str(i))] = dp.shift(sj_test.loc[:,(f)],v)
#
predictions = []
for est in range(1,20):
    for f,l,t in zip(features,labels,test_features):
    
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
                    
        rgr = RandomForestRegressor(min_samples_split =200)
        
        #Data for a Sample Try
        size = int(f.shape[0]*8/10)
        f_t = f.head(size)
        f_t_test = f.tail(f.shape[0] - size)
        l_t = l.head(size)
        l_t_test = l.tail(l.shape[0] - size)
        
        #Try
        rgr.fit(f_t,np.ravel(l_t))
        p = list(map(int,map(round,rgr.predict(f_t_test))))
        er = mean_absolute_error(l_t_test,p)
        print(er,est)
        
        rgr.fit(f,np.ravel(l));
        prd = list(map(int,map(round,rgr.predict(t))))
        predictions.append(prd)
        
##Sace data
#submission.total_cases = np.concatenate(predictions)
#submission.to_csv(dd.DATA_PATH+"new_random_forest_bin.csv")