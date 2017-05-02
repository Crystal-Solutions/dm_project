# -*- coding: utf-8 -*-
"""
Created on Tue May  2 21:50:57 2017

@author: Janaka
"""


DATA_PATH = ".\\data\\clean\\";
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

dfTrain = pd.read_csv(DATA_PATH+'dengue_features_labels_train.csv')
dfTest = pd.read_csv(DATA_PATH+'dengue_features_test.csv')

#Prepare Column names
colNames = dfTrain.columns.values
featureColNames = np.delete(colNames,np.argwhere(colNames=='total_cases' ))
featureColNames = np.delete(featureColNames,np.argwhere(featureColNames=='week_start_date' ))
print(colNames)
[ print(i,end="\t") for i in featureColNames]

#Make X,Y for traing
X = dfTrain.loc[:,featureColNames].values
y = dfTrain['total_cases'].values        
XTest = dfTest.loc[:,featureColNames].values    

#Make regression
rng = np.random.RandomState(1)
regr_1 = DecisionTreeRegressor(max_depth=4)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)

#fit to data
regr_1.fit(X, y)
regr_2.fit(X, y)



                  

y_1 = list(map(round, regr_1.predict(XTest)))
y_2 = list(map(round,regr_2.predict(XTest)))