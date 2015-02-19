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

# NEW Date Variables 
df['hour_cer'] = df['date'] % 100
df['day_cer'] = (df['date'] - df['hour_cer']) / 100
df.sort(['ID', 'date'], inplace = True)

# TRIMMING 
df=df[df['code']==1]
df[(df['stimulus']=='1'])| df['stimulus']=='E'|df['tarrif']=='A')]