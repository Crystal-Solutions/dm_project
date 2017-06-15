# -*- coding: utf-8 -*-
"""
Created on Sun May 14 02:28:56 2017

@author: Janaka
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

#Data related constants
DATA_PATH = "./data/original_data/";
FILE_NAMES = "sj.csv","iq.csv"
DATA_FINISH = ['2008-04-22','2010-06-25']
TEST_START = ['2008-04-29','2010-07-02']
COL_NAMES = ["city", "TimeIndex", "year", "weekofyear", "week_start_date", "ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw", "precipitation_amt_mm", "reanalysis_air_temp_k", "reanalysis_avg_temp_k", "reanalysis_dew_point_temp_k", "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k", "reanalysis_precip_amt_kg_per_m2", "reanalysis_relative_humidity_percent", "reanalysis_sat_precip_amt_mm", "reanalysis_specific_humidity_g_per_kg", "reanalysis_tdtr_k", "station_avg_temp_c", "station_diur_temp_rng_c", "station_max_temp_c", "station_min_temp_c", "station_precip_mm", "total_cases"]
FEATURE_NAMES = [ "ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw", "precipitation_amt_mm", "reanalysis_air_temp_k", "reanalysis_avg_temp_k", "reanalysis_dew_point_temp_k", "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k", "reanalysis_precip_amt_kg_per_m2", "reanalysis_relative_humidity_percent", "reanalysis_sat_precip_amt_mm", "reanalysis_specific_humidity_g_per_kg", "reanalysis_tdtr_k", "station_avg_temp_c", "station_diur_temp_rng_c", "station_max_temp_c", "station_min_temp_c", "station_precip_mm"]
#FEATURE_NAMES = [ "precipitation_amt_mm", "reanalysis_air_temp_k", "reanalysis_avg_temp_k", "reanalysis_dew_point_temp_k", "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k", "reanalysis_precip_amt_kg_per_m2", "reanalysis_relative_humidity_percent", "reanalysis_sat_precip_amt_mm", "reanalysis_specific_humidity_g_per_kg", "reanalysis_tdtr_k", "station_avg_temp_c", "station_diur_temp_rng_c", "station_max_temp_c", "station_min_temp_c", "station_precip_mm"]
TARGET_COL = "total_cases"

#Load Data
data = [ pd.read_csv(DATA_PATH+fileName, parse_dates=['week_start_date'], index_col='week_start_date',date_parser=dateparse) for fileName in FILE_NAMES]

#fill NaN with most recent value
for d in data:
     d.fillna(method='ffill', inplace=True)

#Multiple rows to a single line
base_shift = 0
window_size = 1
features = [frame[FEATURE_NAMES].rolling(window=1,center=False).sum()[base_shift:] for frame in data]
#Merge features form weeks to make many features
new_features = []
for f in features:
    features_array = f.values
    new_features_temp = []
    for i in range(len(features_array)-(window_size-1)):
        new_feature_line = []
        for j in range(window_size):
            new_feature_line = new_feature_line+list(features_array[i+j])
        new_features_temp.append(new_feature_line)
    new_features.append(new_features_temp)

#Prepate targets
targets = [frame[TARGET_COL][base_shift+window_size-1:] for frame in data]

data_finish_indexes = [len(targets[i][:DATA_FINISH[i]])-1 for i in range(2)]

print("Predicting")
predicted = []
for i in range(2):    
    
    rng = np.random.RandomState(1)
    regr_1 = DecisionTreeRegressor(max_depth=3)
    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=3),
                              n_estimators=300, random_state=rng)
    clf = MLPRegressor(solver='lbfgs', tol=1e-20, max_iter=30000, alpha=1e-5, hidden_layer_sizes=(20,4), random_state=1)
    
    regr = clf
    
    f = new_features[i]
    t = targets[i]
    
    
    
    #plot details of features
    X_indices = np.arange(len(f[0]))
    selector = SelectPercentile(f_regression, percentile=50)
    selector.fit(f, t)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()
    plt.bar(X_indices, scores, width=.5,
            label=r'Univariate score ($-Log(p_{value})$)', color='blue')
    plt.show()
    
    effective_indexes = []
    for j in range(len(scores)):
        if scores[j]>0.7:
            effective_indexes.append(j)
    
    f_after_removing_features = []
    for line in f:
        new_line = []
        for j in effective_indexes:
            new_line.append(line[j])
        f_after_removing_features.append(new_line)
    
    f = f_after_removing_features
    
    for e_i in effective_indexes:
        print(e_i,FEATURE_NAMES[e_i])
    
    sample_train_features = f[:data_finish_indexes[i]+1][:-150]
    sample_train_targets = t[:DATA_FINISH[i]][:-150]
    sample_test_features = f[:data_finish_indexes[i]+1][-150:]
    sample_test_targets = t[:DATA_FINISH[i]][-150:]
    
    #fit a sample program
    regr.fit(sample_train_features, sample_train_targets)
    
    sample_test_predict_target = regr.predict(sample_test_features)
    sample_test_predict_target = [ pVal if pVal>0 else 0 for pVal in sample_test_predict_target]
#    plt.plot(sample_test_predict_target,color='blue',label='Predicted')
#    plt.show()
#    plt.plot(sample_test_targets,color='red',label='Original')
#    plt.show()

    print(mean_absolute_error(sample_test_targets,sample_test_predict_target))


    #Train with all and predict
    regr.fit(f[:data_finish_indexes[i]+1], t[:DATA_FINISH[i]])
    pred = list(map(round,regr.predict(f[data_finish_indexes[i]+1:])))
    predicted.append([ pVal if pVal>0 else 0 for pVal in pred])
    

    