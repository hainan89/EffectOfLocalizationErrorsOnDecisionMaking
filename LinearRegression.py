# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 08:26:00 2017

@author: Casey-NS
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import PyRegressionKits as RegKit
import pickle

import statsmodels.api as sm
import statsmodels.formula.api as smf

#dataSet, dataTrain, dataTest = RegKit.loadData()
dataFull, dataTrain, dataTest = RegKit.loadData()
dataSet = dataFull[dataFull['cep'] >= 1] 
dataSet = dataSet[dataSet['Y'] >= 1]
dataSet = dataSet[dataSet['R'] >= 1]

def xFun(x):
    return 1 / (1 + x)

rP = smf.ols('Precision ~ xFun(cep) + xFun(R) + xFun(Y)', data=dataSet).fit()
rR = smf.ols('Recall ~ xFun(cep) + xFun(R) + xFun(Y)', data=dataSet).fit()

print(rP.summary())
print(rR.summary())

'''
get regression performance evaluation features
'''
from sklearn.metrics import mean_squared_error, mean_absolute_error
yTrue = dataSet['Precision']
yPred = rP.fittedvalues
MSE = mean_squared_error(yTrue, yPred)
RMSE = np.sqrt(MSE)
MAE = mean_absolute_error(yTrue, yPred)
print(MSE, RMSE, MAE)

'''
residual plot
'''
fig, ax = plt.subplots(2, 2)
ax[0, 0].scatter(dataSet['Precision'], rP.resid, color=[0,0,1,0.2])
ax[0, 0].set_xlabel('Precision')
ax[0, 0].set_ylabel('Residual')
ax[0, 0].grid(True)

ax[0, 1].scatter(dataSet['Recall'], rR.resid, color=[0,0,1,0.2])
ax[0, 1].set_xlabel('Recall')
ax[0, 1].set_ylabel('Residual')
ax[0, 1].grid(True)

ax[1, 0].scatter(dataSet['Precision'], rP.fittedvalues, color=[0,0,1,0.2])
ax[1, 0].plot([0, 1], [0, 1], linewidth=5, color = 'red')
ax[1, 0].set_xlabel('Precision')
ax[1, 0].set_ylabel('Fitted Precision')
ax[1, 0].grid(True)

ax[1, 1].scatter(dataSet['Recall'], rR.fittedvalues, color=[0,0,1,0.2])
ax[1, 1].plot([0, 1], [0, 1], linewidth=5, color = 'red')
ax[1, 1].set_xlabel('Recall')
ax[1, 1].set_ylabel('Fitted Recall')
ax[1, 1].grid(True)

