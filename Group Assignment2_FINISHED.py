from __future__ import division
from pandas import Series, DataFrame
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np 
import os
import matplotlib.pyplot as plt

main_dir = "/Users/lexiyang/Desktop/data"

# ADVANCED PATHING ---------------------------------
root = main_dir + "/cooked/"
paths = [root + v for v in os.listdir(root) if v.startswith("File")]

#IMPORT DATA & Stacking-----------------------------------------------------------
missing=['.','NA','NULL',' ','-','999999999']
list_of_dfs = [pd.read_table(v, names = ['ID', 'time', 'kwh'], sep = " ", na_values= missing) for v in paths]
df_stack = pd.concat (list_of_dfs, ignore_index = True)
del list_of_dfs

df_assign = pd.read_csv(root + "SME and Residential allocations.csv", na_values= missing, usecols = range(0,4))
df_assign.columns = ['ID', 'code', 'tariff', 'stimulus']

# MERGING------------------------------------------------------------------------
df = pd.merge(df_stack, df_assign)

# NEW Date Variables ---------------------------------------------------------------
df['hour_cer'] = df['time'] % 100
df['day_cer'] = (df['time'] - df['hour_cer']) / 100
df.sort(['ID', 'time'], inplace = True)

# TRIMMING  ---------------------------------------------------------------
df = df[df['code'] == 1]
df = df[((df['tariff'] == 'E') & (df['stimulus'] == 'E')) | ((df['tariff'] == 'A') & (df['stimulus'] == '1'))]

# TIME VARIABLE CREATION AND TIME SERIES CORRECTION ---------------------------
df_time = pd.read_csv(root + "timeseries_correction.csv", na_values = missing, header = 0, parse_dates = [1], usecols = [1,2,3,4,9,10])
df = pd.merge(df, df_time)

# We can also try:
# df_time = pd.read_csv(root + "timeseries_correction.csv", na_values = missing, header = 0, parse_dates = [1], usecols = [1,2,9,10])
## ADD/DROP VARIABLES ---------------------------
# df['year'] = df['date'].apply(lambda x: x.year)
# df['month'] = df['date'].apply(lambda x: x.month)
# df['day'] = df['date'].apply(lambda x: x.day)

# MONTHLY AGGREGATION --------------------
grp_month = df.groupby(['year', 'month', 'ID', 'tariff'])
agg_month = grp_month['kwh'].sum()

# reset the index (multilevel at the moment)
agg_month = agg_month.reset_index() # drop the multi-index
grp1_month = agg_month.groupby(['year', 'month', 'tariff'])

# Get separate sets of treatment and control values by date
trt_month = {(k[0],k[1]): agg_month.kwh[v].values for k, v in grp1_month.groups.iteritems() if k[2] == 'A'}
ctrl_month = {(k[0],k[1]): agg_month.kwh[v].values for k, v in grp1_month.groups.iteritems() if k[2] == 'E'}

# create dataframes of this information
keys_month = trt_month.keys()
tstats_month = DataFrame([(k, np.abs(ttest_ind(trt_month[k],ctrl_month[k], equal_var=False)[0])) for k in keys_month],
    columns =['ym', 'tstats_month'])
pvals_month = DataFrame([(k, (ttest_ind(trt_month[k],ctrl_month[k], equal_var=False)[1])) for k in keys_month],
    columns =['ym', 'pvals_month'])
t_p_month = pd.merge(tstats_month, pvals_month)

## sort and reset _index  ---------------------------------------------------------------
t_p_month.sort(['ym'], inplace=True) # inplace = True to change the values
t_p_month.reset_index(inplace=True, drop=True)
# t_p_month = t_p.dropna(axis = 0, how = 'any')  # drop any missing values in tstats and pvals

# PLOTTING ----------------------
fig1 = plt.figure() #initialize plot
ax1 = fig1.add_subplot(2,1,1) # (row, columns, reference) two rows, one column, first plot
ax1.plot(t_p_month['tstats_month'])
ax1.axhline(2, color='r', linestyle ='--')
ax1.axvline(6, color='g', linestyle ='--')
ax1.set_title('t-stats over-time (monthly)')

ax2 = fig1.add_subplot(2,1,2) # (row, columns, reference) two rows, one column, second plot
ax2.plot(t_p_month['pvals_month'])
ax2.axhline( 0.05, color='r', linestyle ='--')
ax2.axvline(6, color='g', linestyle ='--')
ax2.set_title('p-values over-time (monthly)')

# DAILY AGGREGATION --------------------
grp_day = df.groupby(['date', 'ID', 'tariff'])
agg_day = grp_day['kwh'].sum()

# reset the index (multilevel at the moment)  ---------------------------------------------------------------
agg_day = agg_day.reset_index() # drop the multi-index
grp1_day = agg_day.groupby(['date', 'tariff'])

# Get separate sets of treatment and control values by date  ---------------------------------------------------------------
trt_day = {k[0]: agg_day.kwh[v].values for k, v in grp1_day.groups.iteritems() if k[1] == 'A'}
ctrl_day= {k[0]: agg_day.kwh[v].values for k, v in grp1_day.groups.iteritems() if k[1] == 'E'}

# create dataframes of this information ---------------------------------------------------------------
keys_day = trt_day.keys()
tstats_day = DataFrame([(k, np.abs(ttest_ind(trt_day[k],ctrl_day[k], equal_var=False)[0])) for k in keys_day],
    columns =['date', 'tstats_day'])
pvals_day = DataFrame([(k, (ttest_ind(trt_day[k],ctrl_day[k], equal_var=False)[1])) for k in keys_day],
    columns =['date', 'pvals_day'])
t_p_day = pd.merge(tstats_day, pvals_day)

## sort and reset _index ---------------------------------------------------------------
t_p_day.sort(['date'], inplace=True) # inplace = True to change the values
t_p_day.reset_index(inplace = True, drop = True)
##t_p_day = t_p_day.dropna(axis = 0, how = 'any')  # drop any missing values in tstats and pvals

# PLOTTING ----------------------
fig3 = plt.figure() #initialize plot
ax3 = fig3.add_subplot(2,1,1) # (row, columns, reference) two rows, one column, first plot
ax3.plot(t_p_day['tstats_day'])
ax3.axhline(2, color='r', linestyle ='--')
ax3.axvline(180, color='g', linestyle ='--')
ax3.set_title('t-stats over-time (daily)')

ax4 = fig3.add_subplot(2,1,2) # (row, columns, reference) two rows, one column, second plot
ax4.plot(t_p_day['pvals_day'])
ax4.axhline( 0.05, color='r', linestyle ='--')
ax4.axvline( 180, color='g', linestyle ='--')
ax4.set_title('p-values over-time (daily)')



