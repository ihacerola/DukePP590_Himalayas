from __future__ import division
from pandas import Series, DataFrame
import statsmodels.api as sm
import pandas as pd
import numpy as np 
import os

main_dir = "/Users/lexiyang/Desktop/data/"


#####################################################################
#                           SECTION 1                               #
#####################################################################
# PATHING ---------------------------------
## root = main_dir + "/task/"

## Using the new data without parsing date
root = main_dir + "3_task_data/"
df_assign= pd.read_csv(root + "allocation_subsamp.csv")

C=df_assign['ID'][(df_assign['tariff'] == 'E') & (df_assign['stimulus'] == 'E')]
A1=df_assign['ID'][(df_assign['tariff'] == 'A') & (df_assign['stimulus'] == '1')]
A3=df_assign['ID'][(df_assign['tariff'] == 'A') & (df_assign['stimulus'] == '3')]
B1=df_assign['ID'][(df_assign['tariff'] == 'B') & (df_assign['stimulus'] == '1')]
B3=df_assign['ID'][(df_assign['tariff'] == 'B') & (df_assign['stimulus'] == '3')]

## SHORTCUT
df_assign['tarstim']=df_assign['tariff'] + df_assign['stimulus']

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

##df2 = pd.read_csv(root + "kwh_redux_pretrail.csv", parse_dates=[2])
df2 = pd.read_csv(root + "kwh_redux_pretrial.csv")
df = pd.merge(df1, df2)

### df['date'] = df['date'].apply(np.datetime64)
##df['year'] = df['date'].apply(lambda x: x.year)
##df['month'] = df['date'].apply(lambda x: x.month)

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

## Set up A1 vs C
dfA1= df4[( df4.vector==1)| (df4.vector==2)]

# GENDERATE DUMMIES FROM QULITATIVE DATA (i.e. categories)
## pd.get_dummies() will make dummy vectors for All "object" or "category" type
dfA1 = pd.get_dummies(dfA1,columns = ['vector'])
dfA1. drop(['vector_1'], axis = 1, inplace = True)

## SET UP THE DATA FOR LOGIT -------
kwh_cols = [v for v in dfA1.columns.values if v.startswith('kwh')]

## SET UP Y, X
yA1 = dfA1['vector_2']
X = dfA1[kwh_cols]
X = sm.add_constant(X)

## LOGIT ----------------
logit_model = sm.Logit(yA1, X)
logit_results = logit_model.fit()
print(logit_results.summary())

## Set up A3 vs C
dfA3= df4[( df4.vector==1)| (df4.vector==3)]

# GENDERATE DUMMIES FROM QULITATIVE DATA (i.e. categories)
## pd.get_dummies() will make dummy vectors for All "object" or "category" type
dfA3 = pd.get_dummies(dfA3,columns = ['vector'])
dfA3. drop(['vector_1'], axis = 1, inplace = True)

## SET UP THE DATA FOR LOGIT -------
kwh_cols = [v for v in dfA3.columns.values if v.startswith('kwh')]

## SET UP Y, X
yA3 = dfA3['vector_3']
X = dfA3[kwh_cols]
X = sm.add_constant(X)

## LOGIT ----------------
logit_model = sm.Logit(yA3, X)
logit_results = logit_model.fit()
print(logit_results.summary())

## Set up B1 vs C
dfB1= df4[( df4.vector==1)| (df4.vector==4)]

# GENDERATE DUMMIES FROM QULITATIVE DATA (i.e. categories)
## pd.get_dummies() will make dummy vectors for All "object" or "category" type
dfB1 = pd.get_dummies(dfB1,columns = ['vector'])
dfB1. drop(['vector_1'], axis = 1, inplace = True)

## SET UP THE DATA FOR LOGIT -------
kwh_cols = [v for v in dfB1.columns.values if v.startswith('kwh')]

## SET UP Y, X
yB1 = dfB1['vector_4']
X = dfB1[kwh_cols]
X = sm.add_constant(X)

## LOGIT ----------------
logit_model = sm.Logit(yB1, X)
logit_results = logit_model.fit()
print(logit_results.summary())

## Set up B3 vs C
dfB3= df4[( df4.vector==1)| (df4.vector==5)]

# GENDERATE DUMMIES FROM QULITATIVE DATA (i.e. categories)
## pd.get_dummies() will make dummy vectors for All "object" or "category" type
dfB3 = pd.get_dummies(dfB3,columns = ['vector'])
dfB3. drop(['vector_1'], axis = 1, inplace = True)

## SET UP THE DATA FOR LOGIT -------
kwh_cols = [v for v in dfB3.columns.values if v.startswith('kwh')]

## SET UP Y, X
yB3 = dfB3['vector_5']
X = dfB3[kwh_cols]
X = sm.add_constant(X)

## LOGIT ----------------
logit_model = sm.Logit(yB3, X)
logit_results = logit_model.fit()
print(logit_results.summary())


## group task 3 practice
from __future__ import division
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import statsmodels.api as sm

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# DEFINE FUNCTIONS -----------------
def ques_recode(srvy):

    DF = srvy.copy()
    import re
    q = re.compile('Question ([0-9]+):.*')
    cols = [unicode(v, errors ='ignore') for v in DF.columns.values]
    mtch = []
    for v in cols:
        mtch.extend(q.findall(v))

    df_qs = Series(mtch, name = 'q').reset_index() # get the index as a variable. basically a column index
    n = df_qs.groupby(['q'])['q'].count() # find counts of variable types
    n = n.reset_index(name = 'n') # reset the index, name counts 'n'
    df_qs = pd.merge(df_qs, n) # merge the counts to df_qs
    df_qs['index'] = df_qs['index'] + 1 # shift index forward 1 to line up with DF columns (we ommited 'ID')
    df_qs['subq'] = df_qs.groupby(['q'])['q'].cumcount() + 1
    df_qs['subq'] = df_qs['subq'].apply(str)
    df_qs.ix[df_qs.n == 1, ['subq']] = '' # make empty string
    df_qs['Ques'] = df_qs['q']
    df_qs.ix[df_qs.n != 1, ['Ques']] = df_qs['Ques'] + '.' + df_qs['subq']

    DF.columns = ['ID'] + df_qs.Ques.values.tolist()

    return df_qs, DF

def ques_list(srvy):

    df_qs, DF = ques_recode(srvy)
    Qs = DataFrame(zip(DF.columns, srvy.columns), columns = [ "recoded", "desc"])[1:]
    return Qs

# df = dataframe of survey, sel = list of question numbers you want to extract free of DVT
def dvt(srvy, sel):

    """Function to select questions then remove extra dummy column (avoids dummy variable trap DVT)"""

    df_qs, DF = ques_recode(srvy)

    sel = [str(v) for v in sel]
    nms = DF.columns

    # extract selected columns
    indx = []
    for v in sel:
         l = df_qs.ix[df_qs['Ques'] == v, ['index']].values.tolist()
         if(len(l) == 0):
            print (bcolors.FAIL + bcolors.UNDERLINE +
            "\n\nERROR: Question %s not found. Please check CER documentation"
            " and choose a different question.\n" + bcolors.ENDC) % v
         indx =  indx + [i for sublist in l for i in sublist]

    # Exclude NAs Rows
    DF = DF.dropna(axis=0, how='any', subset=[nms[indx]])

    # get IDs
    dum = DF[['ID']]
    # get dummy matrix
    for i in indx:
        # drop the first dummy to avoid dvt
        temp = pd.get_dummies(DF[nms[i]], columns = [i], prefix = 'D_' + nms[i]).iloc[:, 1:]
        dum = pd.concat([dum, temp], axis = 1)
        # print dum

        # test for multicollineary

    return dum

def rm_perf_sep(y, X):

    dep = y.copy()
    indep = X.copy()
    yx = pd.concat([dep, indep], axis = 1)
    grp = yx.groupby(dep)

    nm_y = dep.name
    nm_dum = np.array([v for v in indep.columns if v.startswith('D_')])

    DFs = [yx.ix[v,:] for k, v in grp.groups.iteritems()]
    perf_sep0 = np.ndarray((2, indep[nm_dum].shape[1]),
        buffer = np.array([np.linalg.norm(DF[nm_y].values.astype(bool) - v.values) for DF in DFs for k, v in DF[nm_dum].iteritems()]))
    perf_sep1 = np.ndarray((2, indep[nm_dum].shape[1]),
        buffer = np.array([np.linalg.norm(~DF[nm_y].values.astype(bool) - v.values) for DF in DFs for k, v in DF[nm_dum].iteritems()]))

    check = np.vstack([perf_sep0, perf_sep1])==0.
    indx = np.where(check)[1] if np.any(check) else np.array([])

    if indx.size > 0:
        keep = np.all(np.array([indep.columns.values != i for i in nm_dum[indx]]), axis=0)
        nms = [i.encode('utf-8') for i in nm_dum[indx]]
        print (bcolors.FAIL + bcolors.UNDERLINE +
        "\nPerfect Separation produced by %s. Removed.\n" + bcolors.ENDC) % nms

        # return matrix with perfect predictor colums removed and obs where true
        indep1 = indep[np.all(indep[nm_dum[indx]]!=1, axis=1)].ix[:, keep]
        dep1 = dep[np.all(indep[nm_dum[indx]]!=1, axis=1)]
        return dep1, indep1
    else:
        return dep, indep


def rm_vif(X):

    import statsmodels.stats.outliers_influence as smso
    loop=True
    indep = X.copy()
    # print indep.shape
    while loop:
        vifs = np.array([smso.variance_inflation_factor(indep.values, i) for i in xrange(indep.shape[1])])
        max_vif = vifs[1:].max()
        # print max_vif, vifs.mean()
        if max_vif > 30 and vifs.mean() > 10:
            where_vif = vifs[1:].argmax() + 1
            keep = np.arange(indep.shape[1]) != where_vif
            nms = indep.columns.values[where_vif].encode('utf-8') # only ever length 1, so convert unicode
            print (bcolors.FAIL + bcolors.UNDERLINE +
            "\n%s removed due to multicollinearity.\n" + bcolors.ENDC) % nms
            indep = indep.ix[:, keep]
        else:
            loop=False
    # print indep.shape

    return indep


def do_logit(df, tar, stim, D = None):

    DF = df.copy()
    if D is not None:
        DF = pd.merge(DF, D, on = 'ID')
        kwh_cols = [v for v in DF.columns.values if v.startswith('kwh')]
        dum_cols = [v for v in D.columns.values if v.startswith('D_')]
        cols = kwh_cols + dum_cols
    else:
        kwh_cols = [v for v in DF.columns.values if v.startswith('kwh')]
        cols = kwh_cols

    # DF.to_csv("/Users/dnoriega/Desktop/" + "test.csv", index = False)
    # set up y and X
    indx = (DF.tariff == 'E') | ((DF.tariff == tar) & (DF.stimulus == stim))
    df1 = DF.ix[indx, :].copy() # `:` denotes ALL columns; use copy to create a NEW frame
    df1['T'] = 0 + (df1['tariff'] != 'E') # stays zero unless NOT of part of control
    # print df1

    y = df1['T']
    X = df1[cols] # extend list of kwh names
    X = sm.add_constant(X)

    msg = ("\n\n\n\n\n-----------------------------------------------------------------\n"
    "LOGIT where Treatment is Tariff = %s, Stimulus = %s"
    "\n-----------------------------------------------------------------\n") % (tar, stim)
    print msg

    print (bcolors.FAIL +
        "\n\n-----------------------------------------------------------------" + bcolors.ENDC)

    y, X = rm_perf_sep(y, X) # remove perfect predictors
    X = rm_vif(X) # remove multicollinear vars

    print (bcolors.FAIL +
        "-----------------------------------------------------------------\n\n\n" + bcolors.ENDC)

    ## RUN LOGIT
    logit_model = sm.Logit(y, X) # linearly prob model
    logit_results = logit_model.fit(maxiter=10000, method='newton') # get the fitted values
    print logit_results.summary() # print pretty results (no results given lack of obs)


#####################################################################
#                           SECTION 2                               #
#####################################################################

main_dir = main_dir = "/Users/lexiyang/Desktop/data/"
root = main_dir + "3_task_data/"

nas = ['', ' ', 'NA'] # set NA values so that we dont end up with numbers and text
srvy = pd.read_csv(root + 'Smart meters Residential pre-trial survey data.csv', na_values = nas)
df = pd.read_csv(root + 'data_section2.csv')

# list of questions
qs = ques_list(srvy)

# get dummies
sel = [200, 405, 410]
dummies = dvt(srvy, sel)

# run logit, optional dummies
tariffs = [v for v in pd.unique(df['tariff']) if v != 'E']
stimuli = [v for v in pd.unique(df['stimulus']) if v != 'E']
tariffs.sort() # make sure the order correct with .sort()
stimuli.sort()

for i in tariffs:
    for j in stimuli:
        do_logit(df, i, j, D = dummies)


