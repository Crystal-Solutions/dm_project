from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# just for the sake of this blog post!
from warnings import filterwarnings
filterwarnings('ignore')

DATA_PATH = './data/original_data/'
FEATURE_NAMES = [ "ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw", "precipitation_amt_mm", "reanalysis_air_temp_k", "reanalysis_avg_temp_k", "reanalysis_dew_point_temp_k", "reanalysis_max_air_temp_k", "reanalysis_min_air_temp_k", "reanalysis_precip_amt_kg_per_m2", "reanalysis_relative_humidity_percent", "reanalysis_sat_precip_amt_mm", "reanalysis_specific_humidity_g_per_kg", "reanalysis_tdtr_k", "station_avg_temp_c", "station_diur_temp_rng_c", "station_max_temp_c", "station_min_temp_c", "station_precip_mm"]

# load the provided data
def preprocess_data(data_path, labels_path=None,all_featurs = False):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])
    
    # select features we want
    features = ['reanalysis_specific_humidity_g_per_kg', 
                 'reanalysis_dew_point_temp_k', 
                 'station_avg_temp_c', 
                 'station_min_temp_c']
    if all_featurs:
        features = FEATURE_NAMES
    
    df = df[features]
    
    # fill missing values
    df.fillna(method='ffill', inplace=True)

    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df = df.join(labels)
    
    # separate san juan and iquitos
    sj = df.loc['sj']
    iq = df.loc['iq']
    
    if labels_path:
        sj_l,sj_f = sj['total_cases'],sj[features]
        iq_l,iq_f = iq['total_cases'],iq[features]
        
        sj_f = pd.ewma(sj_f,span=1)
        iq_f = pd.ewma(iq_f,span=1)
        
        sj_f.insert(sj_f.shape[-1],'total_cases',sj_l.values)
        iq_f.insert(iq_f.shape[-1],'total_cases',iq_l.values)
        
        sj = sj_f[5:]
        iq = iq_f[5:]
    else: 
        sj = pd.ewma(sj,span=1)[5:]
        iq = pd.ewma(iq,span=3)[5:]

    
    return sj, iq


sj_train, iq_train = preprocess_data(DATA_PATH+'dengue_features_train.csv',
                                    labels_path=DATA_PATH+"dengue_labels_train.csv")

#split
sj_train_subtrain = sj_train.head(800)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)

iq_train_subtrain = iq_train.head(400)
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)





#Negative binomial model 
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf

def get_best_model(train, test):
    # Step 1: specify the form of the model
    model_formula = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "station_min_temp_c + " \
                    "station_avg_temp_c" 
    
    grid = 10 ** np.arange(-8, -3, dtype=np.float64)
                    
    best_alpha = []
    best_score = 1000
        
    # Step 2: Find the best hyper parameter, alpha
    for alpha in grid:
        model = smf.glm(formula=model_formula,
                        data=train,
                        family=sm.families.NegativeBinomial(alpha=alpha))

        results = model.fit()
        predictions = results.predict(test).astype(int)
        score = eval_measures.meanabs(predictions, test.total_cases)

        if score < best_score:
            best_alpha = alpha
            best_score = score

    print('best alpha = ', best_alpha)
    print('best score = ', best_score)
            
    # Step 3: refit on entire dataset
    full_dataset = pd.concat([train, test])
    model = smf.glm(formula=model_formula,
                    data=full_dataset,
                    family=sm.families.NegativeBinomial(alpha=best_alpha))

    fitted_model = model.fit()
    return fitted_model
    
sj_best_model = get_best_model(sj_train_subtrain, sj_train_subtest)
iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest)
sj_train['fitted'] = sj_best_model.fittedvalues
iq_train['fitted'] = iq_best_model.fittedvalues



#After training binomial, a svm t odetect outbreakers###################################
from sklearn import svm
sj_all_f, iq_all_f = preprocess_data(DATA_PATH+'dengue_features_train.csv',
                                     all_featurs=True)

sj_y = np.where(sj_train['fitted']*2<sj_train['total_cases'],1,0)
clf_sj = svm.SVC()
clf_sj.fit(sj_all_f[:], sj_y[:])
sj_train['svm'] = clf_sj.predict(sj_all_f)
sj_train['svm_added_fitted'] = np.where(sj_train['svm']==1,sj_train['fitted']*2,0)

iq_y = np.where(iq_train['fitted']*2<iq_train['total_cases'],1,0)
clf_iq = svm.SVC()
clf_iq.fit(iq_all_f[:], iq_y[:])
iq_train['svm'] = clf_iq.predict(iq_all_f)
iq_train['svm_added_fitted'] = np.where(iq_train['svm']==1,iq_train['fitted']*2,0)


#Plot all
figs, axes = plt.subplots(nrows=2, ncols=1)

# plot sj
sj_train.fitted.plot(ax=axes[0], label="Predictions")
sj_train.total_cases.plot(ax=axes[0], label="Actual")
sj_train.svm_added_fitted.plot(ax=axes[0], label="SVM")

# plot iq
iq_train.fitted.plot(ax=axes[1], label="Predictions")
iq_train.total_cases.plot(ax=axes[1], label="Actual")
iq_train.svm_added_fitted.plot(ax=axes[1], label="SVM")

plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
plt.legend()





#Final Prediction###################################################################
sj_test, iq_test = preprocess_data(DATA_PATH+'dengue_features_test_added_lines_from_test.csv')

sj_predictions = sj_best_model.predict(sj_test).astype(int)
iq_predictions = iq_best_model.predict(iq_test).astype(int)

submission = pd.read_csv(DATA_PATH+"submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv(DATA_PATH+"out1.csv")