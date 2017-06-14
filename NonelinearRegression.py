# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:28:28 2017

@author: Casey-NS
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import PyRegressionKits as RegKit
import pickle

import statsmodels.api as sm
import statsmodels.formula.api as smf

def simModelR(x, YVal, RVal, cep):
    
    # model 1
    a = x[0]
    b = x[1]
    c = x[2]
#    p = 1 / ( 1 + np.e ** (-( (a * (1/cep)  ) * YVal + ( b * (1/cep) + c ) )) ) + d #+ ((e*cep) * (cep) )
    p = 1 / ( 1 + np.e ** (-( (a * (1/cep)  ) * YVal + ( b  ) )) ) + c
    return p

def residFunR(x, YVal, RVal, cep, PVal):
    p_ = simModelR(x, YVal, RVal, cep)
    return PVal - p_

def simModelP(x, YVal, RVal, cep):
    a = x[0]
    b = x[1]
    c = x[2]
    p = - np.power(1 / (1 + a * 1/(cep) ), b * YVal + c ) + 1
    
    return p

def residFunP(x, YVal, RVal, cep, PVal):
    p_ = simModelP(x, YVal, RVal, cep)
    return PVal - p_

dataFull, dataTrain, dataTest = RegKit.loadData()
dataSet = dataFull[dataFull['cep'] >= 1] 
dataSet = dataSet[dataSet['Y'] >= 1]
dataSet = dataSet[dataSet['R'] >= 1]
from scipy.optimize import least_squares

x0 = [1, 1, 1, 1, 1, 1]
resP = least_squares(residFunP, x0, args=(dataSet['Y'], dataSet['R'], dataSet['cep'], 
                                          dataSet['Precision']),
                                          f_scale = 0.1, loss='soft_l1')
pP = simModelP(resP.x, dataSet['Y'], dataSet['R'], dataSet['cep'])

x0 = [1, 1, 1, 1, 1, 1]
resR = least_squares(residFunR, x0, args=(dataSet['Y'], dataSet['R'], dataSet['cep'], 
                                          dataSet['Recall']),
                                          f_scale = 0.1, loss='soft_l1')
pR = simModelR(resR.x, dataSet['Y'], dataSet['R'], dataSet['cep'])

residP = dataSet['Precision'] - pP
residR = dataSet['Recall'] - pR

plt.figure()
plt.hist(residR, 100)

plt.figure()
plt.scatter(dataSet['Y'], dataSet['Recall'], color = 'blue')
plt.scatter(dataSet['Y'], pR, color = 'red')


'''
get regression performance evaluation features
'''
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
yTrue = dataSet['Precision']
yPred = pP
MSE = mean_squared_error(yTrue, yPred)
RMSE = np.sqrt(MSE)
MAE = mean_absolute_error(yTrue, yPred)
R2 = r2_score(yTrue, yPred)
print(MSE, RMSE, MAE, R2)


'''
show the estimated value vs real value
'''
fig, ax = plt.subplots(2, 2)
ax[0, 0].scatter(dataSet['Precision'], residP, color=[0,0,1,0.2])
ax[0, 0].set_xlabel('Precision')
ax[0, 0].set_ylabel('Residual')
ax[0, 0].grid(True)

ax[0, 1].scatter(dataSet['Recall'], residR, color=[0,0,1,0.2])
ax[0, 1].set_xlabel('Recall')
ax[0, 1].set_ylabel('Residual')
ax[0, 1].grid(True)

ax[1, 0].scatter(dataSet['Precision'], pP, color=[0,0,1,0.2])
ax[1, 0].plot([0, 1], [0, 1], linewidth=5, color = 'red')
ax[1, 0].set_xlabel('Precision')
ax[1, 0].set_ylabel('Fitted Precision')
ax[1, 0].grid(True)

ax[1, 1].scatter(dataSet['Recall'], pR, color=[0,0,1,0.2])
ax[1, 1].plot([0, 1], [0, 1], linewidth=5, color = 'red')
ax[1, 1].set_xlabel('Recall')
ax[1, 1].set_ylabel('Fitted Recall')
ax[1, 1].grid(True)

