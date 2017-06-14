# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:09:09 2017

@author: Casey-NS
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import PyRegressionKits as RegKit
import pickle

import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy.optimize import least_squares

def simModel(x, YVal, RVal, cep):
    
    # model 1
    a = x[0]
    b = x[1]
    c = x[2]
    d = x[3]
    e = x[4]
    f = x[5]
    p = 1 / ( 1 + np.e ** (-( (a * (1/cep) + b ) * YVal + ( c * (1/cep) + d ) )) ) + ((e*cep) * (cep) )
    
    #model 2
#    a = x[0]
#    b = x[1]
#    c = x[2]
#    d = x[3]
#    p = - np.power(1 / (1 + a * 1/cep), b * YVal + c * RVal + d) + 1
    
    return p

def residFun(x, YVal, RVal, cep, PVal):
    p_ = simModel(x, YVal, RVal, cep)
    return PVal - p_

dataFull, dataTrain, dataTest = RegKit.loadData()
#dataSet = dataFull#[dataFull['cep'] == 4.1]

xList = []
for cep in np.arange(1, 5, 0.1):
    cep = np.around([cep], decimals = 1)
    dataSet = dataFull[dataFull['cep'] >= 1] 
    dataSet = dataSet[dataSet['Y'] >= 1]
    dataSet = dataSet[dataSet['R'] >= 1]
    
    print(len(dataSet))
    
    '''
    subset illustration
    '''
    #dataR01 = dataSet[dataSet['cep'] == 1]
    #dataR05 = dataSet[dataSet['cep'] == 2]
    #dataR10 = dataSet[dataSet['cep'] == 3]
    #
    #plt.figure()
    #plt.scatter(dataR10['Y'], dataR10['Precision'], c='blue')
    ##plt.figure()
    #plt.scatter(dataR01['Y'], dataR01['Precision'], c='red')
    ##plt.figure()
    #plt.scatter(dataR05['Y'], dataR05['Precision'], c='green')
    
    
    
    testItem = 'Recall'
    
    x0 = [1, 1, 1, 1, 1, 1, 1, 1]
    res_2 = least_squares(residFun, x0, args=(dataSet['Y'], dataSet['R'],dataSet['cep'],
                                              dataSet[testItem]),
                                              f_scale = 0.1, loss='soft_l1')
    
#    xList.append({'a':res_2.x[0],'b':res_2.x[1],'c':res_2.x[2]})
        
    p = simModel(res_2.x, dataSet['Y'], dataSet['R'], dataSet['cep'])
    
    plt.figure()
    plt.scatter(dataSet['Y'], dataSet[testItem], c='blue')
    plt.scatter(dataSet['Y'], p, c='red')
    plt.title("Predicted vs Real {0}".format(testItem))
    
    r = residFun(res_2.x, dataSet['Y'], dataSet['R'], dataSet['cep'], dataSet[testItem])

    break

'''
show parameters vs cep
'''
#xListDf = pd.DataFrame(xList)
#plt.figure()
#plt.scatter(np.arange(1, 5, 0.1), xListDf['a'])


'''
plot the residuals
'''
plt.figure()
plt.scatter(dataSet[testItem], r)
plt.title("{0} vs residuals".format(testItem))

plt.figure()
plt.scatter(dataSet[testItem], p)
plt.plot([0, 1], [0, 1], linewidth=5, color = 'red')
plt.title("True vs Predicted")

print(res_2.x)

'''
model test for sigmoid
'''

#x = np.arange(0, 10, 0.1)
#def sigmoidY(d, x):
#    y = 1 / (1 + d ** -(x ) )
#    return y
#
#plt.figure()
#for d in np.arange(2, 7, 1):
#    y = sigmoidY(d, x)
#    colorV = np.random.rand(3,1)
#    plt.scatter(x, y, label = 'd {0}'.format(d), c=colorV)
#plt.legend()
#
#
#showItem = testItem
#plt.figure()
#for cep in np.arange(0.1, 4.9, 1):
##    cep = 1.1
#    dataSet = dataFull[dataFull['cep'] == cep]
##    p = simModel(res_2.x, dataSet['Y'], dataSet['R'], dataSet['cep'])
#    
#    colorV = np.random.rand(3,1)
#    plt.scatter(dataSet['Y'], dataSet[showItem], label='cep {0}'.format(cep),
#                c=colorV)
#    
##    p = sigmoidY(dataSet['Y'])
##    plt.scatter(dataSet['Y'], p, label='cep {0}'.format(cep), marker='o',
##                c='red')
#    
##    break
#    
#plt.title(showItem)
#plt.legend()







