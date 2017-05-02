# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 01:16:41 2017

@author: Janaka
"""

DATA_PATH = "J:\\Raw\\CS\\Sem7\\DM\\dengi_data\\";
import pandas as pd
from sklearn import datasets, svm, metrics
import math
import matplotlib.pyplot as plt




def yw_to_i(y,w):
    y-=1990
    w-=1
    return y*53+w

def i_to_yw(i):
    return (i//53+1990,(i%53+1))


df_f = pd.read_csv(DATA_PATH+'dengue_features_train.csv')
df_t = pd.read_csv(DATA_PATH+'dengue_labels_train.csv')
df_f_test = pd.read_csv(DATA_PATH+'dengue_features_test.csv')
cities = ['sj','iq']
feature_names = 'city','year','weekofyear','ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw','precipitation_amt_mm','reanalysis_air_temp_k','reanalysis_avg_temp_k','reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k','reanalysis_min_air_temp_k','reanalysis_precip_amt_kg_per_m2','reanalysis_relative_humidity_percent','reanalysis_sat_precip_amt_mm','reanalysis_specific_humidity_g_per_kg','reanalysis_tdtr_k','station_avg_temp_c','station_diur_temp_rng_c','station_max_temp_c','station_min_temp_c','station_precip_mm'
#feature_names = 'city','year','weekofyear','week_start_date','ndvi_ne','ndvi_nw','ndvi_se','ndvi_sw','precipitation_amt_mm','reanalysis_air_temp_k','reanalysis_avg_temp_k','reanalysis_dew_point_temp_k','reanalysis_max_air_temp_k','reanalysis_min_air_temp_k','reanalysis_precip_amt_kg_per_m2','reanalysis_relative_humidity_percent','reanalysis_sat_precip_amt_mm','reanalysis_specific_humidity_g_per_kg','reanalysis_tdtr_k','station_avg_temp_c','station_diur_temp_rng_c','station_max_temp_c','station_min_temp_c','station_precip_mm'


#Fill timeline with empty values.. 
MIN_I = yw_to_i(1990,1)
MAX_I = yw_to_i(2014,53)
timeline = {}
for c in cities:
    for i in range(MIN_I,MAX_I+1):
        timeline[(c,i)] = {'features':[], 'target':-1}


max_target = -1
min_target = 10000000
for index, row in df_t.iterrows():
    c,y,w = (row['city'], row['year'], row['weekofyear'])
    i = yw_to_i(y,w)
    target_value = row['total_cases']
    timeline[(c,i)]['target'] = target_value
    if(max_target<target_value):
        max_target = target_value
    if min_target>target_value:
        min_target = target_value
        

for index, row in df_f.iterrows():
    entry = []
    c,y,w = (row['city'], row['year'], row['weekofyear'])
    i = yw_to_i(y,w)
    for f in feature_names:
        entry.append(row[f])
    timeline[(c,i)]['features'] = entry
    

count = 0
for key, value in timeline.items():
    if value['target'] == -1:
        count+=1
        



#fill valuse using the mask
mask = [1,1,1,1,1,1,1]
solved = []
solvedF = []
for key, value in timeline.items():
    c,i = key
    if value['target'] == -1:
        count_of_non_neg = 0
        sum_of_non_neg = 0
        for k in range(i-len(mask)//2, i+1+len(mask)//2):
            if(k<MIN_I or k>MAX_I):
                continue
            j = k-i+len(mask)//2
            t = timeline[(c,k)]['target'] # target value
            if((c,k) in timeline and t!=-1):
                count_of_non_neg+=mask[j]
                sum_of_non_neg+=t*mask[j]
        if count_of_non_neg == 0: 
            value['target'] = 0
        else:
            value['target'] = sum_of_non_neg/count_of_non_neg
        solved.append((count_of_non_neg, sum_of_non_neg,c, i_to_yw(i)))

    if len(value['features'])==0:
        count_of_non_neg = 0
        sums_of_non_neg = [0 for x in feature_names]
        for k in range(i-len(mask)//2, i+1+len(mask)//2):
            if(k<MIN_I or k>MAX_I):
                continue
            j = k-i+len(mask)//2
            features = timeline[(c,k)]['features'] # target value
            if((c,k) in timeline and len(features)!=0):
                count_of_non_neg+=mask[j]
                new_s=[y if isinstance(y, str) 
                else (x + mask[j]*y if math.isnan(y) else 0) for x, y  in zip(sums_of_non_neg, features)]
                sums_of_non_neg = new_s
        if count_of_non_neg == 0: 
            value['features'] = [0 for x in feature_names]
        else:
            value['features'] = [x if isinstance(x, str) else x/count_of_non_neg for x in sums_of_non_neg]
        solvedF.append((count_of_non_neg, sums_of_non_neg,c, i_to_yw(i)))
        

                
       
        

plt.scatter(list(map(lambda x: x[1], timeline.keys())), 
         list(map(lambda x: 1 if timeline[x]['target']==-1 else 0, timeline.keys())))
plt.show()


plt.scatter(list(map(lambda x: x[1], timeline.keys())), 
         list(map(lambda x: 1 if timeline[x]['features']==[] else 0, timeline.keys())))
plt.show()
#features, targets = [],[]
#for index, row in df_f.iterrows():
#    row_data = []
#    skip = False;
#    for colname in df_f:
#        if(colname!='city' and colname!='week_start_date' and  not math.isnan(row[colname])):
#            row_data.append(row[colname])
#    features.append(row_data);
#    targets.append(cases_from_week[(row['city'], row['year'], row['weekofyear'])]);
#

#Train Classifier
#classifier = svm.SVC(gamma=0.001)
#classifier.fit(features, targets);