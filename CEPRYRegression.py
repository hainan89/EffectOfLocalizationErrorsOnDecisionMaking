# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 20:11:55 2017

@author: Casey-NS
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import pickle
from PrecisionRecallOnCPFWL import *
import os


def CEP2CPF(cep):
    cpf = np.array([[0, 0.5, 0.93, 1.0],[0, cep, cep * 2, cep* 3]])
    return cpf

def RY2WarningLevel(rVal, yVal):
    return {"R":rVal, "Y":(rVal + yVal), "G":(rVal + yVal)}

if __name__ == '__main__':
    
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    cep2LowerLimitSeg = float(input("input the lower limit of the cep value: "))
    cep2UpperLimitSeg = float(input("input the upper limit of the cep value: "))
    
    cep2LowerLimit = np.floor(cep2LowerLimitSeg)
    if cep2LowerLimit == 0:
        cep2LowerLimit = 0.1
        
    cep2UpperLimit = np.ceil(cep2UpperLimitSeg)

    
    folderName = "cep-{0}-{1}".format(cep2LowerLimit, cep2UpperLimit)
    folderPath = "./{0}".format(folderName)
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)
    
#    simulationDataSet = []

#    testI = int(round((cep2LowerLimitSeg - cep2LowerLimit) / 0.1) * 49 * 49)
#    subFolder = 0
#    os.mkdir("./{0}/{1}".format(folderName, subFolder))
    
#    for cep in np.arange(0.1, 10, 0.01):
    cepRange = np.arange(int(cep2LowerLimitSeg * 10), int(cep2UpperLimitSeg * 10), 1) / 10
    for cep in cepRange:
        for rval in np.arange(1, 50, 1) / 10:
            for yval in np.arange(1, 50, 1) / 10:
                
                cep = round(cep, 1)
                rval = round(rval, 1)
                yval = round(yval, 1)
                
#                cep, rval, yval = 0.1, 0.5, 0.4
                testI = (round(cep - cep2LowerLimit, 1) / 0.1) * 49 * 49
                testI = testI + (round(rval - 0.1, 1) / 0.1) * 49 + round(yval * 10)
                
#                if np.mod(testI, 100) == 0:
                subFolder = int(np.ceil(testI / 100))
                subFilderPath = "./{0}/{1}".format(folderName, subFolder)
                if not os.path.exists(subFilderPath):
                    os.mkdir(subFilderPath)
                        
                print("cep, rval, yval", cep, rval, yval)
#                if np.mod(testI, 2) == 0:
                fn1 = './{0}/{1}/cpf-warningLevel-{2}-{3}-{4}.pkl'.format(folderName, subFolder, cep, rval, yval)
                
                if os.path.isfile(fn1):
                    print('Data File Existed.')
                    continue

                cpf = CEP2CPF(cep)
                wl = RY2WarningLevel(rval, yval)
                sVal = getPrecisionRecall(cpf, wl)

                with open(fn1, 'wb') as f1:                     # open file with write-mode
                    picklestring = pickle.dump(sVal, f1)
#                    simulationDataSet =[]
                    print("----Save Data File")


#    listVal = []
#    for i in np.arange(2, 10, 2):
#        pathDataFName = "cpf-warningLevel-{0}.pkl".format(i)
#        with open(pathDataFName, 'rb') as f:
#            val = pickle.load(f)   # read file and build object
#            listVal.append(val)    