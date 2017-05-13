# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:50:10 2017

@author: Janaka
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

#Data related constants
DATA_PATH = ".\\data\\clean\\";
FILE_NAMES = "sj.csv","iq.csv"
DATA_FINISH = ['2008-04-22','2010-06-25']
TEST_START = ['2008-04-29','2010-07-02']
COL_NAMES = ["city", "TimeIndex", "year", "weekofyear", "week_start_date", "ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw", "precipitation_amt_mm", "reanalysis_air_temp_k", "reanalysis_avg_temp_k", "reanalysis_dew_point_temp_k", "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k", "reanalysis_precip_amt_kg_per_m2", "reanalysis_relative_humidity_percent", "reanalysis_sat_precip_amt_mm", "reanalysis_specific_humidity_g_per_kg", "reanalysis_tdtr_k", "station_avg_temp_c", "station_diur_temp_rng_c", "station_max_temp_c", "station_min_temp_c", "station_precip_mm", "total_cases"]
FEATURE_NAMES = ["city", "ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw", "precipitation_amt_mm", "reanalysis_air_temp_k", "reanalysis_avg_temp_k", "reanalysis_dew_point_temp_k", "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k", "reanalysis_precip_amt_kg_per_m2", "reanalysis_relative_humidity_percent", "reanalysis_sat_precip_amt_mm", "reanalysis_specific_humidity_g_per_kg", "reanalysis_tdtr_k", "station_avg_temp_c", "station_diur_temp_rng_c", "station_max_temp_c", "station_min_temp_c", "station_precip_mm"]
TARGET_COL = "total_cases"

#Load Data
data = [ pd.read_csv(DATA_PATH+fileName, parse_dates=['week_start_date'], index_col='week_start_date',date_parser=dateparse) for fileName in FILE_NAMES]


features = [frame[FEATURE_NAMES].shift(2).rolling(window=1,center=False).mean()[5:] for frame in data]
targets = [frame[TARGET_COL][5:] for frame in data]

#print(features)
#print(features[0])
#print(targets[0])


predicted = []
for i in range(2):    
    
    regr = DecisionTreeRegressor(max_depth=3)

     
    
    f = features[i]
    t = targets[i]
    
    sample_train_features = f[:DATA_FINISH[i]][:-200]
    sample_train_targets = t[:DATA_FINISH[i]][:-200]
    sample_test_features = f[:DATA_FINISH[i]][-200:]
    sample_test_targets = t[:DATA_FINISH[i]][-200:]
    
    #fit a sample program
    regr.fit(sample_train_features, sample_train_targets)
    
    sample_test_predict_target = regr.predict(sample_test_features)

    plt.plot(sample_test_predict_target,color='blue',label='Predicted')
    plt.show()
    plt.plot(sample_test_targets,color='red',label='Original')
    plt.show()

    print(mean_absolute_error(sample_test_targets,sample_test_predict_target))


    #Train with all and predict
    regr.fit(features[i][:DATA_FINISH[i]], targets[i][:DATA_FINISH[i]])
    pred = list(map(round,regr.predict(features[i][TEST_START[i]:])))
    predicted.append(pred)
    
