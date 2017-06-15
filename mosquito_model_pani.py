from __future__ import print_function
from __future__ import division
import pandas as pd
from statsmodels.tools import eval_measures
import statsmodels.formula.api as smf
import numpy as np
import statsmodels.api as sm

from matplotlib import pyplot as plt

# just for the sake of this blog post!
from warnings import filterwarnings
filterwarnings('ignore')

def shift(df,n):
    df = df.shift(n)
    df.fillna(method='bfill', inplace=True)
    return df

def preprocess_data(data_path, labels_path=None):
    # load data and set index to city, year, weekofyear
    df = pd.read_csv(data_path, index_col=[0, 1, 2])

    df['reanalysis_relative_humidity_percent_shifted'] = shift(df['reanalysis_relative_humidity_percent'], 8)
    df['precipitation_amt_mm_shifted'] = shift(df['precipitation_amt_mm'], 8)
    df['precipitation_amt_mm_shifted2'] = shift(df['precipitation_amt_mm'],2)
    df['reanalysis_precip_amt_kg_per_m2_shifted'] = shift(df['reanalysis_precip_amt_kg_per_m2'], 8)
    df['reanalysis_dew_point_temp_k_shifted'] = shift(df['reanalysis_dew_point_temp_k'], 11)

    #reanalysis_dew_point_temp_k
    #reanalysis_max_air_temp_k
    #reanalysis_min_air_temp_k

    # select features we want
    features_sj = ['reanalysis_specific_humidity_g_per_kg',
                'reanalysis_dew_point_temp_k',
                'station_avg_temp_c',
                'station_max_temp_c',
                   'reanalysis_relative_humidity_percent',
                   'reanalysis_relative_humidity_percent_shifted',
                   'precipitation_amt_mm_shifted',
                   'reanalysis_precip_amt_kg_per_m2_shifted',
                   'reanalysis_dew_point_temp_k_shifted',
                   'precipitation_amt_mm_shifted2']

    features_iq = ['reanalysis_specific_humidity_g_per_kg',
                   'reanalysis_dew_point_temp_k',
                   'station_avg_temp_c',
                   'station_min_temp_c',
                   'reanalysis_min_air_temp_k']

    df_sj = df[features_sj]
    df_iq = df[features_iq]

    #print('features: ', df.head(5))
    #print('labels  : ', sj_train_labels.shape)

    # fill missing values
    df_sj.fillna(method='ffill', inplace=True)
    df_iq.fillna(method='ffill', inplace=True)
    #print('features: ', df.head(5))
    # add labels to dataframe
    if labels_path:
        labels = pd.read_csv(labels_path, index_col=[0, 1, 2])
        df_sj = df_sj.join(labels)
        df_iq = df_iq.join(labels)

    # separate san juan and iquitos
    sj = df_sj.loc['sj']
    iq = df_iq.loc['iq']

    return sj, iq

sj_train, iq_train = preprocess_data('data-processed/my_train.csv',labels_path="data-processed/dengue_labels_train.csv")

#print(sj_train.describe())

sj_train_subtrain = sj_train.head(800)
sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)

iq_train_subtrain = iq_train.head(400)
iq_train_subtest = iq_train.tail(iq_train.shape[0] - 400)

sj_correlations = sj_train_subtrain.corr()
iq_correlations = iq_train_subtrain.corr()

# San Juan
'''(sj_correlations
     .total_cases
     .drop('total_cases') # don't compare with myself
     .sort_values(ascending=False)
     .plot
     .barh())'''

# Iquitos
'''(iq_correlations
     .total_cases
     .drop('total_cases') # don't compare with myself
     .sort_values(ascending=False)
     .plot
     .barh())'''

def get_best_model(train, test, model_formula ):
    # Step 1: specify the form of the model

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

model_formula_sj = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "station_max_temp_c + " \
                    "reanalysis_relative_humidity_percent + "\
                    "reanalysis_relative_humidity_percent_shifted + "\
                    "precipitation_amt_mm_shifted + "\
                    "reanalysis_precip_amt_kg_per_m2_shifted + "\
                    "reanalysis_dew_point_temp_k_shifted + "\
                    "precipitation_amt_mm_shifted2 + "\
                   "station_avg_temp_c"

model_formula_iq = "total_cases ~ 1 + " \
                    "reanalysis_specific_humidity_g_per_kg + " \
                    "reanalysis_dew_point_temp_k + " \
                    "station_min_temp_c + " \
                    "reanalysis_min_air_temp_k +" \
                    "station_avg_temp_c"

print('....................For SJ.............................')
sj_best_model = get_best_model(sj_train_subtrain, sj_train_subtest, model_formula_sj)
print('\n')
print('....................For IQ.............................')
iq_best_model = get_best_model(iq_train_subtrain, iq_train_subtest, model_formula_iq)
print('\n')

figs, axes = plt.subplots(nrows=2, ncols=1)

# plot sj
sj_train['fitted'] = sj_best_model.fittedvalues
sj_train.fitted.plot(ax=axes[0], label="Predictions")
sj_train.total_cases.plot(ax=axes[0], label="Actual")

# plot iq
iq_train['fitted'] = iq_best_model.fittedvalues
iq_train.fitted.plot(ax=axes[1], label="Predictions")
iq_train.total_cases.plot(ax=axes[1], label="Actual")

#plt.suptitle("Dengue Predicted Cases vs. Actual Cases")
#plt.legend()
#plt.show()

sj_test, iq_test = preprocess_data('data-processed/dengue_features_test.csv')

sj_predictions = sj_best_model.predict(sj_test).astype(int)
iq_predictions = iq_best_model.predict(iq_test).astype(int)

submission = pd.read_csv("data-processed/submission_format.csv",
                         index_col=[0, 1, 2])

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("data-processed/test4.csv")
