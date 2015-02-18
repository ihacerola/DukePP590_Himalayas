from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np 
import os

main_dir="/Users/newlife-cassie/Desktop"

# ADVANCED PATHING----------------------------------------------------
root = main_dir + "/Group_Assignment_Data/"
paths = [root + v for v in os.listdir(root) if v.startswith("File")]
paths1=[root +"File"+str(v)+".txt" for v in range (1,7)]

#IMPORT DATA-----------------------------------------------------------
missing=['.','NA','NULL',' ','0','-','999999999']
list_of_dfs = [pd.read_table(v, names = ['ID', 'date', 'kwh'], sep = " ",na_values= missing) for v in paths]

df_assign = pd.read_csv(root + "SME and Residential allocations.csv",na_values= missing, usecols = [0,1,2,3,4])