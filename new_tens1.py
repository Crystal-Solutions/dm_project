# -*- coding: utf-8 -*-
"""
Created on Wed May 24 20:06:00 2017

@author: Janaka
"""

import pandas as pd
import numpy as np
import dengue_data as dd
import dengue_processing as dp
from matplotlib import pyplot as plt

ef_lst = dd.BM_FEATURE_NAMES
cityNames = dd.CITY_NAMES
train,labels, test, submission = dd.getData(ef_lst)

train = train.join(labels)

sj_train, iq_train  = train.loc['sj'],train.loc['iq']
sj_test, iq_test = test.loc['sj'],test.loc['iq']
sj_train['station_avg_temp_c_2'] = dp.shift(sj_train['station_avg_temp_c'],2)
sj_test['station_avg_temp_c_2'] = dp.shift(sj_test['station_avg_temp_c'],2)




#split
sj_train_subtrain = sj_train.head(800)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)

iq_train_subtrain = iq_train.head(400)
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)

sj_best_model = dp.getBMNegBinomailModel(sj_train_subtrain, sj_train_subtest,dp.DEFAULT_MODEL+' + station_avg_temp_c_2')
iq_best_model = dp.getBMNegBinomailModel(iq_train_subtrain, iq_train_subtest)



figs, axes = plt.subplots(nrows=2, ncols=1)

# plot sj
sj_train['fitted'] = sj_best_model.fittedvalues
sj_train.fitted.plot(ax=axes[0], label="Predictions")
sj_train.total_cases.plot(ax=axes[0], label="Actual")

# plot iq
iq_train['fitted'] = iq_best_model.fittedvalues
iq_train.fitted.plot(ax=axes[1], label="Predictions")
iq_train.total_cases.plot(ax=axes[1], label="Actual")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()


sj_predictions = sj_best_model.predict(sj_test).astype(int)
iq_predictions = iq_best_model.predict(iq_test).astype(int)


submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv(dd.DATA_PATH+"new_bin.csv")