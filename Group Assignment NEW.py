from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np 
import os

main_dir="/Users/newlife-cassie/Desktop"

# ADVANCED PATHING 
root = main_dir + "/Group_Assignment_Data/"
paths = [root + v for v in os.listdir(root) if v.startswith("File")]
paths1=[root +"File"+str(v)+".txt" for v in range (1,7)]

#IMPORT DATA
missing=['.','NA','NULL',' ','0','-','999999999']
list_of_dfs = [pd.read_table(v, names = ['ID', 'date', 'kwh'], sep = " ",na_values= missing,skiprows=100,nrows=200, ) for v in paths]

df_assign = pd.read_csv(root + "SME and Residential allocations.csv",na_values= missing, usecols = [0,1,2,3,4])

# MISSING DATA
df. dropna(how="all") #drop ROWS with ALL missing values

#STACKING AND MERGING  
df_stack = pd.concat(list_of_dfs, ignore_index = True) 
del list_of_dfs
df = pd.merge(df_stack, df_assign)

#CHECK DAYLIGHT SAVING DATES
df['tt']=df['date']% 100
df1 = df[df['tt'] <49]
df[df.date==45203] 

# DROP DUPLICATES
t_b=df.duplicated()
b_t=df.duplicated(take_last=True)
unique=~(t_b|b_t) #true from top to bottom or bottom to top returns true 
unique=~t_b & ~b_t

