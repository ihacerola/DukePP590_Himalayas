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
missing=['.','NA','NULL',' ','0','-','999999999']
list_of_dfs = [pd.read_table(v, names = ['ID', 'date', 'kwh'], sep = " ", na_values= missing) for v in paths]
df_stack = pd.concat (list_of_dfs, ignore_index = True)
del list_of_dfs

df_assign = pd.read_csv(root + "SME and Residential allocations.csv",na_values= missing, usecols = [0,1,2,3])
df_assign.columns = ['ID', 'code', 'tariff', 'stimulus']

# MERGING------------------------------------------------------------------------
df = pd.merge(df_stack, df_assign)

# NEW Date Variables 
df['hour_cer'] = df['date'] % 100
df['day_cer'] = (df['date'] - df['hour_cer']) / 100
df.sort(['ID', 'date'], inplace = True)

# TRIMMING 
df = df[df['code'] == 1]
df = df[(df['stimulus'] == 'E') | ((df['stimulus'] == '1') & (df['tariff'] == 'A'))]

# Examing the data
df1 = df[(df['tariff'] == 'A') & (df['stimulus'] != '1')] # checking if there are people only assigned tariff A but not stimulus 1 "bi-monthy bill"
df2 = df[(df['tariff'] != 'A') & (df['stimulus'] == '1')] # checking if there are people only stimulus 1 but not tariff A
df3 = df[(df['tariff'] == 'E') & (df['stimulus'] != 'E')] 
df4 = df[(df['tariff'] != 'E') & (df['stimulus'] == 'E')] # checking if people are assigned in control group are in the control in both treatment
df5 = df[df['tariff'] == 'A']

# Set up data 
grp = df.groupby(['tariff', 'date'])

# Get separate sets of treatment and control values by date
trt = {k[1]: df.kwh[v].values for k, v in grp.groups.iteritems() if k[0] == 'A'}
ctrl = {k[1]: df.kwh[v].values for k, v in grp.groups.iteritems() if k[0] == 'E'}

# create dataframes of this information
keys = trt.keys()
tstats = DataFrame([(k, np.abs(ttest_ind(trt[k],ctrl[k], equal_var=False)[0])) for k in keys],
    columns =['date', 'tstats'])
pvals = DataFrame([(k, np.abs(ttest_ind(trt[k],ctrl[k], equal_var=False)[1])) for k in keys],
    columns =['date', 'pvals'])
t_p = pd.merge(tstats, pvals)

## sort and reset _index
t_p.sort(['date'], inplace=True) # inplace = True to change the values
t_p.reset_index(inplace=True, drop=True)
t_p = t_p.dropna(axis = 0, how = 'any')  # drop any missing values in tstats and pvals

# PLOTTING ----------------------
fig1 = plt.figure() #initialize plot
ax1 = fig1.add_subplot(2,1,1) # (row, columns, reference) two rows, one column, first plot
ax1.plot(t_p['tstats'])
ax1.axhline(2, color='r', linestyle ='--')
ax1.set_title('t-stats over-time')

ax2 = fig1.add_subplot(2,1,2) # (row, columns, reference) two rows, one column, second plot
ax2.plot(t_p['pvals'])
ax2.axhline( 0.05, color='r', linestyle ='--')
ax2.set_title('p-values over-time')