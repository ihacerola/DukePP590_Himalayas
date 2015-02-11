from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os

main_dir = "/Users/lexiyang/Desktop/data"

# ADVANCED PATHING ---------------------------------
root = main_dir + "/cooked/"
paths = [root + v for v in os.listdir(root) if v.startswith("File")]

# IMPORT DATA -----------------------------------
list_of_dfs = [pd.read_table(v, names = ['ID', 'date', 'kwh'], sep = " ",skiprows = 100, nrows = 200) for v in paths]
len(list_of_dfs)
type(list_of_dfs)
type(list_of_dfs[0])

## ASSIGNMENT DATA
df_assign = pd.read_csv(root + "SME and Residential allocations.csv",usecols = [0,1,2,3,4])

# STACK AND MERGE ----------------
df_stack= pd.concat(list_of_dfs, ignore_index = True)
del list_of_dfs
df = pd.merge(df_stack, df_assign)

## CLEANING DATA ------------------
list = ['66949', '66950', '29849', '29850']
