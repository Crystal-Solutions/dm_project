# -*- coding: utf-8 -*-
"""
Created on Mon May 15 07:12:11 2017

@author: Shanika Ediriweera
"""
import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

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
