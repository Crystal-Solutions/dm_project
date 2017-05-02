# -*- coding: utf-8 -*-
"""
Created on Mon May  1 22:12:07 2017

@author: Janaka
"""

DATA_PATH = "J:\\Raw\\CS\\Sem7\\DM\\dengi_data\\";
import pandas as pd
from sklearn import datasets, svm, metrics
import math
import matplotlib.pyplot as plt


#mthods
def yw_to_i(y,w):
    y-=1990
    w-=1
    return y*53+w

def i_to_yw(i):
    return ((i//53)+1990,((i%53)+1))

MIN_I = yw_to_i(1990,1)
MAX_I = yw_to_i(2014,53)

df_f = pd.read_csv(DATA_PATH+'dengue_features_train.csv')
df_t = pd.read_csv(DATA_PATH+'dengue_labels_train.csv')
df_f_test = pd.read_csv(DATA_PATH+'dengue_features_test.csv')
cities = ['sj','iq']
feature_names = 'city','year','weekofyear','ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw','precipitation_amt_mm','reanalysis_air_temp_k','reanalysis_avg_temp_k','reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k','reanalysis_min_air_temp_k','reanalysis_precip_amt_kg_per_m2','reanalysis_relative_humidity_percent','reanalysis_sat_precip_amt_mm','reanalysis_specific_humidity_g_per_kg','reanalysis_tdtr_k','station_avg_temp_c','station_diur_temp_rng_c','station_max_temp_c','station_min_temp_c','station_precip_mm'
#feature_names = 'city','year','weekofyear','week_start_date','ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw','precipitation_amt_mm','reanalysis_air_temp_k','reanalysis_avg_temp_k','reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k','reanalysis_min_air_temp_k','reanalysis_precip_amt_kg_per_m2','reanalysis_relative_humidity_percent','reanalysis_sat_precip_amt_mm','reanalysis_specific_humidity_g_per_kg','reanalysis_tdtr_k','station_avg_temp_c','station_diur_temp_rng_c','station_max_temp_c','station_min_temp_c','station_precip_mm'


pd.da
features = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

features = {}
for f in feature_names:
    features[f] = [float('NaN') for i in range(MIN_I,MAX_I+1)]

for i in range(MIN_I,MAX_I+1):
    