from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np 
import os

main_dir="/Users/newlife-cassie/Desktop"

# ADVANCED PATHING---------------------------------------------------------------
root = main_dir + "/Group_Assignment_Data/"
paths = [root + v for v in os.listdir(root) if v.startswith("File")]

#IMPORT DATA & Stacking-----------------------------------------------------------
missing=['.','NA','NULL',' ','0','-','999999999']
df = pd.concat ([pd.read_table(v, names = ['ID', 'date', 'kwh'], sep = " ",na_values= missing) for v in paths], ignore_index=True)

df_assign = pd.read_csv(root + "SME and Residential allocations.csv",na_values= missing, usecols = [0,1,2,3,4])
df_assign.columns = ['ID', 'code', 'tariff', 'stimulus', 'sme']

# MERGING------------------------------------------------------------------------
df = pd.merge(df, df_assign)

# NEW DATE VARIABLES------------------------------------------------------------- 
df['hour_cer'] = df['date'] % 100
df['day_cer'] = (df['date'] - df['hour_cer']) / 100
df.sort(['ID', 'date'], inplace = True)

# TRIMMING-----------------------------------------------------------------------
df=df[df['code']==1]
df[(df['stimulus']=='1')| (df['stimulus']=='E')|(df['tariff']=='A')]

# GROUPING-----------------------------------------------------------------------

##Testing if there is any item with stimulus equals 1 but tariff equals E
df1=df[df.tariff=='E']
df2=df1[df.stimulus ==1] #This dataframe is empty, meaning that there is no tarrif E in stimulus 1 group 
# Treatment / control depends on stimulus value 
# stimulus = E --> control 
# stimulus = 1 --> treatment 

groups1=df.groupby(['stimulus','date'])

# ASSIGN TREATMENT AND CONTROL---------------------------------------------------

from scipy.stats import ttest_ind 
from scipy.special import stdtr

trt={k[1]:df.kwh[v].values for k,v in groups1.groups.iteritems() if k[5]=='1'}
crt={k[1]:df.kwh[v].values for k,v in groups1.groups.iteritems() if k[5]=='E'}

# T-TEST & CREATING DATAFRAMES---------------------------------------------------
keys=trt.keys()

tstats=DataFrame([(k,np.abs(ttest_ind(trt[k],crt[k],equal_var=False)[0])) for k in keys],columns=['date','tstats'])
pvals=DataFrame([(k,np.abs(ttest_ind(trt[k],crt[k],equal_var=False)[1])) for k in keys],columns=['date','pvals'])
t_p=pd.merge(tstats,pvals)

t_p.sort(['date'], inplace=True) 
t_p.reset_index(inplace=True,drop=True)

# PLOTTING-----------------------------------------------------------------------
fig1=plt.figure()
ax1=fig1.add_subplot(2,1,1)
ax1.plot(t_p['tstats'])
ax1.axhline(2,color='r',linestyle='--')
ax1.set_title('tstats over time')