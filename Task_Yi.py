from __future__ import division
from pandas import Series, DataFrame
import statsmodels.api as sm
import pandas as pd
import numpy as np 
import os

main_dir = "/Users/lexiyang/Desktop/data"

# ADVANCED PATHING ---------------------------------
root = main_dir + "/task/"
df_assign= pd.read_csv(root + "allocation_subsamp.csv")

C=df_assign['ID'][(df_assign['tariff'] == 'E') & (df_assign['stimulus'] == 'E')]
A1=df_assign['ID'][(df_assign['tariff'] == 'A') & (df_assign['stimulus'] == '1')]
A3=df_assign['ID'][(df_assign['tariff'] == 'A') & (df_assign['stimulus'] == '3')]
B1=df_assign['ID'][(df_assign['tariff'] == 'B') & (df_assign['stimulus'] == '1')]
B3=df_assign['ID'][(df_assign['tariff'] == 'B') & (df_assign['stimulus'] == '3')]

np.random.seed(seed=1789)
C=np.random.choice(C, size=300, replace=False, p=None)
dfC = DataFrame(C,columns = ['ID'])
dfC['vector']=1

A1=np.random.choice(A1, size=150, replace=False, p=None)
dfA1 = DataFrame(A1,columns = ['ID'])
dfA1['vector']=2

A3=np.random.choice(A3, size=150, replace=False, p=None)
dfA3 = DataFrame(A3,columns = ['ID'])
dfA3['vector']=3

B1=np.random.choice(B1, size=50, replace=False, p=None)
dfB1 = DataFrame(B1,columns = ['ID'])
dfB1['vector']=4

B3=np.random.choice(B3, size=50, replace=False, p=None)
dfB3 = DataFrame(B3,columns = ['ID'])
dfB3['vector']=5

df1 = pd.concat([dfC,dfA1,dfA3,dfB1,dfB3], ignore_index = True)
df_assign = pd.merge(df_assign, df1)

#df1 = DataFrame(np.concatenate((C, A1, A3, B1, B3)), columns = ['ID'])

df2 = pd.read_csv(root + "kwh_redux_pretrail.csv", parse_dates=[2])
df = pd.merge(df1, df2)

### df['date'] = df['date'].apply(np.datetime64)
df['year'] = df['date'].apply(lambda x: x.year)
df['month'] = df['date'].apply(lambda x: x.month)

# MONTHLY AGGREGATION --------------------
grp_month = df.groupby(['year', 'month', 'ID', 'vector'])
df3 = grp_month['kwh'].sum().reset_index()

# From 'long' to 'wide'

# create column names for wide data
# create string names and denote consumption and month
# use ternery expression "[true-expr(x) if condition else false-exp(x) for x in list]
# df['day_str'] = ['0' + str(v) if v < 10 else str(v) for v in df1['day']] # add '0' to <10

df3['month'] = ['0' + str(v) if v < 10 else str(v) for v in df3['month']] # add '0' to <10
df3['kwh_ym'] ='kwh_' + df3.year.apply(str) + "_" + df3.month.apply(str)

# . pivot: aka long to wide
df3_piv =df3.pivot ('ID', 'kwh_ym', 'kwh')

# clean up for making things pretty
df3_piv.reset_index(inplace=True) # this makes panid its own variable
df3_piv.columns.name = None

# MERGE TIME invariant data -------
df4 = pd.merge(df_assign, df3_piv) # this attacthing order looks better
del df3_piv, df_assign


# GENDERATE DUMMIES FROM QULITATIVE DATA (i.e. categories)
## pd.get_dummies() will make dummy vectors for All "object" or "category" type

df5 = pd.get_dummies(df4,columns = ['vector'])
df5. drop(['vector_1'], axis = 1, inplace = True)

## SET UP THE DATA FOR LOGIT -------
kwh_cols = [v for v in df5.columns.values if v.startswith('kwh')]

## get cols
## vector_cols = [v for v in df5.columns.values if v.startswith('vector')]

## SET UP Y, X
### I have a hard time understanding the control group here------------
## Treatment A1 vs Non A1 or C
y2 = df5['vector_2']
y3 = df5['vector_3']
y4 = df5['vector_4']
y5 = df5['vector_5']

X = df5[kwh_cols]
X = sm.add_constant(X)

## LOGIT ----------------
logit_model = sm.Logit(y2, X)
logit_results = logit_model.fit()
print(logit_results.summary())

logit_model = sm.Logit(y3, X)
logit_results = logit_model.fit()
print(logit_results.summary())


logit_model = sm.Logit(y4, X)
logit_results = logit_model.fit()
print(logit_results.summary())

logit_model = sm.Logit(y5, X)
logit_results = logit_model.fit()
print(logit_results.summary())