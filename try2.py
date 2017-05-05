# -*- coding: utf-8 -*-
"""
Created on Wed May  3 01:26:35 2017

@author: Janaka
"""

def add_history(lst):
    shift = 5
    ret= []
    for i,row in enumerate(lst[shift:]):
        newRow = list(row[:])
        for j in range(i-shift,i):
            for item in lst[j]:
                newRow.append(item)
        ret.append(newRow)
    return ret


DATA_PATH = ".\\data\\clean\\";
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


dfs = pd.read_csv(DATA_PATH+'sj.csv'),pd.read_csv(DATA_PATH+'iq.csv')
dfMeta = pd.read_csv(DATA_PATH+'meta.csv')

testStartIndexes = 936,520



#Prepare Column names
colNames = dfs[0].columns.values
featureColNames = np.delete(colNames,np.argwhere(colNames=='total_cases' ))
featureColNames = np.delete(featureColNames,np.argwhere(featureColNames=='week_start_date' ))
print(colNames)
[ print(i,end="\t") for i in featureColNames]

results = []
#Train model for each city
for i,df in enumerate(dfs):
    #Make X,Y for traing
    X = df.loc[:testStartIndexes[i]-1,featureColNames].values
    y = df.loc[5:testStartIndexes[i]-1,'total_cases'].values        
    XTest = df.loc[testStartIndexes[i]-5:,featureColNames].values 
                  
    newX = add_history(X)
    newXTest = add_history(XTest)
    
    #Make regression
    rng = np.random.RandomState(1)
    regr_1 = DecisionTreeRegressor(max_depth=4)
    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                              n_estimators=300, random_state=rng)
    
    #fit to data
    regr_1.fit(newX, y)
    regr_2.fit(newX, y)
    
    
    y_1 = list(map(round, regr_1.predict(newXTest)))
    y_2 = list(map(round,regr_2.predict(newXTest)))
    results.append((y_1, y_2))
