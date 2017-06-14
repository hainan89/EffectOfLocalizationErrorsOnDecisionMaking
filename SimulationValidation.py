# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:15:02 2017

@author: Casey-NS
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import scipy
#import pickle
from PrecisionRecallOnCPFWL import *

def CEP2CPF(cep):
    cpf = np.array([[0, 0.5, 0.93, 1.0],[0, cep, cep * 2, cep* 3]])
    return cpf


'''
# offset sampling simulation
'''
plt.figure()
markerList = ['s','o', '*']
colorList = ['red', 'blue', 'green']
flagI = 0
for cep in np.arange(5, 20, 5):
    cpf = CEP2CPF(cep)
    
    areaRef = scipy.integrate.simps(cpf[0,:], cpf[1,:], dx = 0.01)
    
    areaDifSimuList = []
    for sampleSize in np.arange(10, 1000, 50):
        offsetSimu = []
        for i in np.arange(sampleSize):
            rp, rVal = getRwithCPF(cpf)
            offsetSimu.append({'rp':rp, 'rVal':rVal})
            
        offsetSimuDf = pd.DataFrame(offsetSimu)
        offsetSimuDf.sort_values(['rVal'], inplace = True)

        areaSimu = scipy.integrate.simps(offsetSimuDf['rp'], 
                                         offsetSimuDf['rVal'], dx = 0.01)
        areaDifSimuList.append(areaSimu / areaRef)
    plt.plot(np.arange(10, 1000, 50), areaDifSimuList, marker=markerList[flagI], 
             color = colorList[flagI],
             label = 'CEP:{0}'.format(cep))
    flagI = flagI + 1
plt.legend()
plt.grid(True)
plt.xlabel('Random Sample Times')
plt.ylabel('Simulation Coverage Ratio')


'''
angle sampling times
'''
xa, ya = 0, 0
xb, yb = 0, 0
sampleEfficient = []
sampleEfficientVar = []
sampleEfficientDistribute = []
for angleSampleTime in np.arange(10, 400, 50):
    aveRef = []
    for testI in np.arange(100):
        ra = np.random.rand() * 5 + 1
        rb = np.random.rand() * 5 + 1
        distReal = np.random.rand() * 10 + 10
        xb = distReal
        
        setaA = np.random.rand(angleSampleTime) * (2 * np.pi)
        setaB = np.random.rand(angleSampleTime) * (2 * np.pi)
        
        xa1 = ra * np.cos(setaA) + xa
        ya1 = ra * np.sin(setaA) + ya
        
        xb1 = rb * np.cos(setaB) + xb
        yb1 = rb * np.sin(setaB) + yb
        
        d2points = np.sqrt((xa1 - xb1) ** 2 + (ya1 - yb1) ** 2)
        d2points = np.round(d2points)
        measuredD = np.unique(d2points)
        measuredArea = scipy.integrate.simps(measuredD,np.arange(len(measuredD)))
        
        minD = distReal - ra - rb
        maxD = distReal + ra + rb
        refArea = (minD + maxD) * (maxD - minD + 1) / 2
        areaR = measuredArea / refArea
        aveRef.append(areaR)
    sampleEfficient.append(np.mean(aveRef))
    sampleEfficientDistribute.append(aveRef)
#    sampleEfficientVar.append(np.std(aveRef))
    
plt.figure()
plt.plot(np.arange(10, 400, 50), sampleEfficient, marker='*', label='Mean Ratio')
#plt.errorbar(np.arange(10, 360, 50), sampleEfficient, yerr=sampleEfficientVar)
plt.boxplot(sampleEfficientDistribute, positions = np.arange(10, 400, 50),
            widths = 10)
plt.xlim(0, 400)
plt.legend()
plt.xlabel("Sampling Times")
plt.ylabel("Simulation Coverage Ratio")





