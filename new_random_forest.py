# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 21:03:48 2017

@author: Janaka
"""

import pandas as pd
import numpy as np
import dengue_data as dd
import dengue_processing as dp
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor

ef_lst = dd.getFeatureNames() #All features
label_col = dd.LABEL_COLUMN_NAME
cityNames = dd.CITY_NAMES
train,labels, test, submission = dd.getData(ef_lst)


features  = train.loc['sj'],train.loc['iq']
labels = labels.loc['sj'],labels.loc['iq']
test_features = test.loc['sj'],test.loc['iq']

#Shift 
sj_featurs = features[0]
sj_test = test_features[0]
sj_featurs['station_avg_temp_c2'] = dp.shift(sj_featurs['station_avg_temp_c'],4)
sj_test['station_avg_temp_c2'] = dp.shift(sj_test['station_avg_temp_c'],4)





predictions = []
for f,l,t in zip(features,labels,test_features):
    rgr = RandomForestRegressor(min_samples_split =400,n_estimators=200)
    
    #Data for a Sample Try
    size = int(f.shape[0]*5/6)
#    print(size,f.shape[0])
    f_t = f.head(size)
    f_t_test = f.tail(f.shape[0] - size)
    l_t = l.head(size)
    l_t_test = l.tail(l.shape[0] - size)
    
    #Try
    rgr.fit(f_t,np.ravel(l_t))
    prd = rgr.predict(f_t_test)
    print(mean_absolute_error(l_t_test,prd))
    
    rgr.fit(f,np.ravel(l))
    p = list(map(int,map(round,rgr.predict(t))))
    predictions.append(p)
    
    
#Sace data
submission.total_cases = np.concatenate(predictions)
submission.to_csv(dd.DATA_PATH+"new_random_forest_bin.csv")