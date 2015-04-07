# DukePP590_Himalayas
1. Except A1, there is no significant difference between the treatment (A3, B1, and B3) and control group (EE). The sample is balanced in terms of pre-treatment electricity consumption, gender, internet access, and living conditions. This fits our assumption of random sampling. 
In the A1EE logit model, there is significant difference between treatment (A1) and control group (EE) in electricity consumption in July and December 2009, as well as gender at 5% level. 

-----------------------------------------------------------------
LOGIT where Treatment is Tariff = A, Stimulus = 1
-----------------------------------------------------------------

Optimization terminated successfully.
         Current function value: 0.605244
         Iterations 5
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      T   No. Observations:                  371
Model:                          Logit   Df Residuals:                      360
Method:                           MLE   Df Model:                           10
Date:                Tue, 07 Apr 2015   Pseudo R-squ.:                 0.03537
Time:                        15:30:36   Log-Likelihood:                -224.55
converged:                       True   LL-Null:                       -232.78
                                        LLR p-value:                   0.08701
===============================================================================
                  coef    std err          z      P>|z|      [95.0% Conf. Int.]
-------------------------------------------------------------------------------
const          -1.1816      0.384     -3.073      0.002        -1.935    -0.428
kwh_2009_07     0.0065      0.003      2.258      0.024         0.001     0.012
kwh_2009_08    -0.0035      0.002     -1.621      0.105        -0.008     0.001
kwh_2009_09    -0.0015      0.003     -0.552      0.581        -0.007     0.004
kwh_2009_10 -3.486e-05      0.003     -0.012      0.990        -0.006     0.005
kwh_2009_11    -0.0024      0.002     -1.003      0.316        -0.007     0.002
kwh_2009_12     0.0027      0.001      2.036      0.042      9.89e-05     0.005
D_200_2         0.5055      0.235      2.151      0.031         0.045     0.966
D_405_2        -0.1996      0.268     -0.743      0.457        -0.726     0.327
D_410_2         0.4354      0.329      1.325      0.185        -0.209     1.079
D_410_3         0.5417      0.396      1.367      0.172        -0.235     1.318
===============================================================================


-----------------------------------------------------------------
LOGIT where Treatment is Tariff = A, Stimulus = 3
-----------------------------------------------------------------

Optimization terminated successfully.
         Current function value: 0.606011
         Iterations 5
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      T   No. Observations:                  364
Model:                          Logit   Df Residuals:                      353
Method:                           MLE   Df Model:                           10
Date:                Tue, 07 Apr 2015   Pseudo R-squ.:                 0.01819
Time:                        15:30:36   Log-Likelihood:                -220.59
converged:                       True   LL-Null:                       -224.68
                                        LLR p-value:                    0.6117
===============================================================================
                  coef    std err          z      P>|z|      [95.0% Conf. Int.]
-------------------------------------------------------------------------------
const          -1.1101      0.410     -2.707      0.007        -1.914    -0.306
kwh_2009_07     0.0030      0.003      1.081      0.280        -0.002     0.009
kwh_2009_08    -0.0030      0.002     -1.481      0.139        -0.007     0.001
kwh_2009_09     0.0014      0.003      0.497      0.619        -0.004     0.007
kwh_2009_10     0.0003      0.003      0.098      0.922        -0.005     0.006
kwh_2009_11     0.0016      0.002      0.669      0.504        -0.003     0.006
kwh_2009_12    -0.0018      0.001     -1.233      0.218        -0.005     0.001
D_200_2         0.0272      0.236      0.115      0.908        -0.435     0.490
D_405_2        -0.0859      0.282     -0.304      0.761        -0.640     0.468
D_410_2         0.3862      0.347      1.113      0.266        -0.294     1.067
D_410_3         0.6642      0.416      1.595      0.111        -0.152     1.480
===============================================================================


-----------------------------------------------------------------
LOGIT where Treatment is Tariff = B, Stimulus = 1
-----------------------------------------------------------------

Optimization terminated successfully.
         Current function value: 0.397733
         Iterations 7
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      T   No. Observations:                  295
Model:                          Logit   Df Residuals:                      284
Method:                           MLE   Df Model:                           10
Date:                Tue, 07 Apr 2015   Pseudo R-squ.:                 0.04227
Time:                        15:30:36   Log-Likelihood:                -117.33
converged:                       True   LL-Null:                       -122.51
                                        LLR p-value:                    0.4097
===============================================================================
                  coef    std err          z      P>|z|      [95.0% Conf. Int.]
-------------------------------------------------------------------------------
const          -2.6035      0.654     -3.978      0.000        -3.886    -1.321
kwh_2009_07  9.531e-05      0.004      0.026      0.980        -0.007     0.007
kwh_2009_08    -0.0019      0.003     -0.667      0.504        -0.007     0.004
kwh_2009_09     0.0008      0.003      0.270      0.788        -0.005     0.007
kwh_2009_10     0.0061      0.004      1.618      0.106        -0.001     0.013
kwh_2009_11    -0.0039      0.003     -1.135      0.256        -0.011     0.003
kwh_2009_12    -0.0008      0.002     -0.384      0.701        -0.005     0.003
D_200_2        -0.1243      0.349     -0.356      0.722        -0.809     0.560
D_405_2         0.3449      0.398      0.867      0.386        -0.435     1.125
D_410_2         0.7497      0.563      1.331      0.183        -0.354     1.853
D_410_3         0.9632      0.648      1.487      0.137        -0.306     2.233
===============================================================================


-----------------------------------------------------------------
LOGIT where Treatment is Tariff = B, Stimulus = 3
-----------------------------------------------------------------

Optimization terminated successfully.
         Current function value: 0.377850
         Iterations 6
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      T   No. Observations:                  290
Model:                          Logit   Df Residuals:                      279
Method:                           MLE   Df Model:                           10
Date:                Tue, 07 Apr 2015   Pseudo R-squ.:                 0.02703
Time:                        15:30:36   Log-Likelihood:                -109.58
converged:                       True   LL-Null:                       -112.62
                                        LLR p-value:                    0.8077
===============================================================================
                  coef    std err          z      P>|z|      [95.0% Conf. Int.]
-------------------------------------------------------------------------------
const          -1.7706      0.570     -3.106      0.002        -2.888    -0.653
kwh_2009_07    -0.0026      0.004     -0.648      0.517        -0.011     0.005
kwh_2009_08    -0.0012      0.003     -0.388      0.698        -0.007     0.005
kwh_2009_09     0.0004      0.004      0.092      0.926        -0.008     0.009
kwh_2009_10    -0.0030      0.004     -0.671      0.502        -0.012     0.006
kwh_2009_11     0.0051      0.004      1.434      0.152        -0.002     0.012
kwh_2009_12    -0.0004      0.002     -0.245      0.807        -0.004     0.003
D_200_2         0.1672      0.370      0.452      0.651        -0.558     0.892
D_405_2        -0.5619      0.441     -1.276      0.202        -1.425     0.301
D_410_2         0.0431      0.514      0.084      0.933        -0.965     1.051
D_410_3        -0.2490      0.662     -0.376      0.707        -1.547     1.049
===============================================================================


2. Benefit: Including all the available survey variables can help us to check if the assignment of treatment and control is randomized or not. In other words, it can help us to check if the two groups are balanced in terms of every possible attribute.
Potential problems: The major problem is collinearity since the available survey variables might be correlated with each other.

3. It would be appropriate to use a subset of survey data when 1) the subset includes all major determinants of treatment assignment, 2) the subset of variables are not correlated with each other, and 3) data are available for this subset. 
