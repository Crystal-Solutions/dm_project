# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:08:37 2017

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

ef_lst = dd.getFeatureNames() #.BM_FEATURE_NAMES +[''] #All features
label_col = dd.LABEL_COLUMN_NAME
cityNames = dd.CITY_NAMES
train,labels, test, submission = dd.getData(ef_lst)


features  = train.loc['sj'],train.loc['iq']
labels = labels.loc['sj'],labels.loc['iq']
test_features = test.loc['sj'],test.loc['iq']

#Shift 
sj_featurs = features[0]
sj_test = test_features[0]


#Correlations
f,l,t  = (features[0].copy(),labels[0],test_features[0])
f['total_cases'] = l
for i in range(1,5):
    for feature in ef_lst:
        f[feature+'_'+str(i)] =  dp.shift(f.loc[:,(feature)],i)
cor = f.corr().total_cases.sort_values(ascending=False)

corDict = {}
for f_name,v in cor.items():
    if f_name[-1] not in '123456789':
        continue;
    f_name,shift = f_name[:-2],int(f_name[-1])
    
    if f_name not in corDict.keys():
        corDict[f_name] = []
    corDict[f_name].append((v,shift))
    
for k,v in corDict.items():
    print(k,max(v,key=lambda x:x[0]))
