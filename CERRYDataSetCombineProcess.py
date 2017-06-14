# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 14:13:56 2017

@author: Casey-NS
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import pickle
import os
import PrecisionRecallOnCPFWL as CPFWL


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
    
fileList = []
def getFiles(rootDir): 
    for lists in os.listdir(rootDir): 
        
        path = os.path.join(rootDir, lists) 
#        print path
        print(len(fileList), path)
        if os.path.isdir(path):
            getFiles(path)
        else:
            fileList.append(path)
      
rootDir = input('Input the root dir: ')
getFiles(rootDir)

with open('fileList-{0}.pkl'.format(rootDir), 'wb') as f0:
    pickle.dump(fileList, f0)
    
#with open('fileList.pkl', 'rb') as f0:
#    fileList = pickle.load(f0)

fileNum = len(fileList)
fileIndexList = np.arange(len(fileList))
np.random.shuffle(fileIndexList)

dataSet = []
'''
set up training and test data set
'''
saveI = 0
dataSetLen = 0
for i in np.arange(fileNum):
    
    progress = i / fileNum
    print(progress, dataSetLen, saveI)
    
    filePath = fileList[fileIndexList[i]]
    
    fName = os.path.basename(filePath)
    tags = fName.split('-')
    cep = float(tags[2])
    
    with open(filePath,'rb') as f0:
        oneData = pickle.load(f0)
    
    R = oneData['warningLevel']['R']
    Y = oneData['warningLevel']['Y']
    
    m = len(oneData['decisionValues'])
    
    for decisionI in np.arange(m):
        if decisionI == 0:
            df = oneData['decisionValues'][decisionI]['df']
        else:
            df = df + oneData['decisionValues'][decisionI]['df']
            
    precision, recall = CPFWL.getPrecisionRecallforOne(df)
    oneRecord = {'cep':cep, 'R':R, 'Y':Y, 'Precision':precision, 'Recall':recall}
    
    dataSet.append(oneRecord)
    dataSetLen = len(dataSet)
    
    if (dataSetLen == 10 ** 4) or (i == (fileNum - 1)):
        # save data
        dataSetDf = pd.DataFrame(dataSet)
        dataSetDf.to_csv('dataSetDf-3In-2Out-{0}-{1}.csv'.format(rootDir, saveI), index_label='index')
        dataSet = []
        saveI = saveI + 1
