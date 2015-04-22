from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

main_dir = '/Users/lexiyang/Desktop/data/task 4/'

#####################################################################
#                           SECTION 0                              #
#####################################################################


# CHANGE WORKING DIRECTORY (wd)
os.chdir(main_dir)
from logit_functions import *

# IMPORT DATA ------------
df = pd.read_csv(main_dir + '14_B3_EE_w_dummies.csv')
df = df.dropna(axis=0, how='any')

# GET TARIFFS ------------
tariffs = [v for v in pd.unique(df['tariff']) if v != 'E']
stimuli = [v for v in pd.unique(df['stimulus']) if v != 'E']
tariffs.sort()
stimuli.sort()

# RUN LOGIT
drop = [v for v in df.columns if v.startswith("kwh_2010")]
df_pretrial = df.drop(drop, axis=1)

for i in tariffs:
    for j in stimuli:
        # dummy vars must start with "D_" and consumption vars with "kwh_"
        logit_results, df_logit = do_logit(df_pretrial, i, j, add_D=None, mc=False)
        
# QUICK MEANS COMPARISON WITH T-TEST BY HAND----------
# create means
grp = df_logit.groupby('tariff')
df_mean = grp.mean().transpose()
df_mean.B - df_mean.E

# do a t-test "by hand"
df_s = grp.std().transpose()
df_n = grp.count().transpose().mean()
top = df_mean['B'] - df_mean['E']
bottom = np.sqrt(df_s['B']**2/df_n['B'] + df_s['E']**2/df_n['E'])
tstats = top/bottom
sig = tstats[np.abs(tstats) > 2]
sig.name = 't-stats'

#####################################################################
#                           SECTION 1                               #
#####################################################################


# IMPORT DATA ------------
df = pd.read_csv(main_dir + 'task_4_kwh_w_dummies_wide.csv')
df = df.dropna(axis=0, how='any')

# GET TARIFFS ------------
tariffs = [v for v in pd.unique(df['tariff']) if v != 'E']
stimuli = [v for v in pd.unique(df['stimulus']) if v != 'E']
tariffs.sort()
stimuli.sort()

# RUN LOGIT
drop = [v for v in df.columns if v.startswith("kwh_2010")]
df_pretrial = df.drop(drop, axis=1)

for i in tariffs:
    for j in stimuli:
        # dummy vars must start with "D_" and consumption vars with "kwh_"
        logit_results, df_logit = do_logit(df_pretrial, i, j, add_D=None, mc=False)

# QUICK MEANS COMPARISON WITH T-TEST BY HAND----------
# create means
grp = df_logit.groupby('tariff')
df_mean = grp.mean().transpose()
df_mean.C - df_mean.E

# do a t-test "by hand"
df_s = grp.std().transpose()
df_n = grp.count().transpose().mean()
top = df_mean['C'] - df_mean['E']
bottom = np.sqrt(df_s['C']**2/df_n['C'] + df_s['E']**2/df_n['E'])
tstats = top/bottom
sig = tstats[np.abs(tstats) > 2]
sig.name = 't-stats'


#####################################################################
#                           SECTION 2                               #
#####################################################################


df_logit['p_hat'] = logit_results.predict()
df_logit['trt'] = 0 + (df_logit['tariff'] == 'C')
df_logit['w'] = np.sqrt(df_logit['trt'] /df_logit['p_hat']+(1 - df_logit['trt'] )/(1 - df_logit['p_hat']))

df_w = df_logit[['ID', 'trt', 'w']]


#####################################################################
#                           SECTION 3                               #
#####################################################################

from fe_functions import *
df_long = pd.read_csv(main_dir + 'task_4_kwh_long.csv')
df = pd.merge(df_long, df_w)

df['TP'] = df['trt'] * df['trial']
df['log_kwh'] = (df['kwh'] + 1).apply(np.log)
# create month string `mo_str` that adds "0" to single digit integers
df['mo_str'] = np.array(["0" + str(v) if v < 10 else str(v) for v in df['month']])
# concatenate to make ym string values
df['ym'] = df['year'].apply(str) + "_" + df['mo_str']

## SET UP Y, X
y = df['log_kwh']
P = df['trial']
TP = df['TP']
w = df['w']
mu = pd.get_dummies(df['ym'], prefix = 'ym').iloc[:, 1:-1]
X = pd.concat([TP, P, mu], axis=1)

ids = df['ID']
y = demean(y, ids)
X = demean(X, ids)

## WITHOUT WEIGHTS
fe_model = sm.OLS(y, X) # linearly prob model
fe_results = fe_model.fit() # get the fitted values
print(fe_results.summary()) # print pretty results (no results given lack of obs)

# WITH WEIGHTS
## apply weights to data
y = y*w # weight each y
nms = X.columns.values # save column names
X = np.array([x*w for k, x in X.iteritems()]) # weight each X value
X = X.T # transpose (necessary as arrays create "row" vectors, not column)
X = DataFrame(X, columns = nms) # update to dataframe; use original names

fe_w_model = sm.OLS(y, X) # linearly prob model
fe_w_results = fe_w_model.fit() # get the fitted values
print(fe_w_results.summary()) # print pretty results (no results given lack of obs)
