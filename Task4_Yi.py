from __future__ import division
import pandas as pd
import numpy as np
import os

main_dir = '/Users/lexiyang/Desktop/data/task 4/'

#####################################################################
#                           SECTION 1                               #
#####################################################################


# CHANGE WORKING DIRECTORY (wd)
os.chdir(main_dir)
from logit_functions import *

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
df_logit['T'] = 0 + (df_logit['tariff'] == 'C')
df_logit['w'] = np.sqrt(df_logit['T'] /df_logit['p_hat']+(1 - df_logit['T'] )/(1 - df_logit['p_hat']))



#####################################################################
#                           SECTION 3                               #
#####################################################################

df_long = pd.read_csv(main_dir + 'task_4_kwh_long.csv')
df_merge = pd.merge(df_long, df_logit)