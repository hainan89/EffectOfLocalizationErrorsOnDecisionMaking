# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 20:23:36 2017
## get precision and recall based on the given cpf and warning level
@author: Casey-NS
"""


import numpy as np
import pandas as pd
import time
import pickle
## get r value basedon cpf
def getRwithCPF(cpf):
    rp = np.random.rand()
    for i in np.arange(cpf.shape[1]):
        if cpf[0, i] <= rp and cpf[0, i+1] >= rp:
            break
        else:
            continue
    rVal = cpf[1, i] + (rp - cpf[0, i]) / (cpf[0, i + 1] - cpf[0, i]) * (cpf[1, i + 1] - cpf[1, i])
    return rp, rVal
 
## test decision result based on the given position and cpf with given warning level
def inRangeProbability(xa, ya, xb, yb, cpf, dRed, dYellow):
    rNum = 1000
    setaNum = 360
    checkNum = rNum * setaNum
    redNum = 0
    yellowNum = 0
    greenNum = 0
    for rt in np.arange(rNum):
        rpa, ra = getRwithCPF(cpf)
        setaA = np.random.rand(setaNum) * (2 * np.pi)
        
        rpb, rb = getRwithCPF(cpf)
        setaB = np.random.rand(setaNum) * (2 * np.pi)
        
        xa1 = ra * np.cos(setaA) + xa
        ya1 = ra * np.sin(setaA) + ya
        
        xb1 = rb * np.cos(setaB) + xb
        yb1 = rb * np.sin(setaB) + yb
        
        d2points = (xa1 - xb1) ** 2 + (ya1 - yb1) ** 2
        
        redNumC = len(d2points[d2points <= (dRed ** 2)])
        greenNumC = len(d2points[d2points > (dYellow ** 2)])
        yellowNumC = len(d2points) - (redNumC + greenNumC)
        
        redNum = redNum + redNumC
        yellowNum = yellowNum + yellowNumC
        greenNum = greenNum + greenNumC
        
    return checkNum, redNum, yellowNum, greenNum

## get real  deciiosn for the given two position with given warning level
def getRealStatus(pa, pb, warningLevel):
    rDistSQ = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
    if rDistSQ <= (warningLevel["R"] ** 2):
        return "R"
    
    if rDistSQ > (warningLevel["R"] ** 2) and rDistSQ <= (warningLevel["Y"] ** 2):
        return "Y"

    if rDistSQ > (warningLevel["Y"] ** 2):
        return "G"
   
        
## get the decision table for tow positions with given cpf and warning level
def cpfWarningLevelTest(cpf, warningLevel, pa, pb):
    
    warningLevelList = list(warningLevel.keys())
    evaluationDf = pd.DataFrame(np.zeros((len(warningLevelList), len(warningLevelList))),
                                columns=warningLevelList, index=warningLevelList)
    
    checkNum, redNum, yellowNum, greenNum = inRangeProbability(pa[0], pa[1], 
                         pb[0], pb[1],
                         cpf,
                         warningLevel["R"], warningLevel["Y"])

    
    # columns are the real status
    realStatus = getRealStatus(pa, pb, warningLevel)
    
    evaluationDf[realStatus]["R"] = evaluationDf[realStatus]["R"] + (redNum / checkNum)
    evaluationDf[realStatus]["Y"] = evaluationDf[realStatus]["Y"] + (yellowNum / checkNum)
    evaluationDf[realStatus]["G"] = evaluationDf[realStatus]["G"] + (greenNum / checkNum)
    
    return evaluationDf
    
## get precision and recall with given decision table
def getPrecisionRecallforOne(df):
    oneTest = df
    tp = oneTest["R"]["R"] + oneTest["R"]["Y"] + oneTest["Y"]["R"] + oneTest["Y"]["Y"]
    fn = oneTest["G"]["R"] + oneTest["G"]["Y"]
    fp = oneTest["R"]["G"] + oneTest["Y"]["G"]
    tn = oneTest["G"]["G"]
    
    precision = tp / (tp + fn)
    recall = tp / (tp + fp)
    
    return precision, recall

## get normalized decision tabel along with the column value
def normalizedDf(df):
    normDf = df
    for oneColumn in df.columns:
        normDf[oneColumn] = normDf[oneColumn] / sum(normDf[oneColumn])
        
    return normDf
    
    
    
## get precision and recall based on the cpf and warning level
def getPrecisionRecall(cpf, warningLevel):

    stepSize = 0.1
    decisionValues = []
    
    rangeLimit = 4 * (warningLevel["Y"] + cpf[1][-1])
    offSetList = np.arange(0, rangeLimit, stepSize)
    
    print("getPrecisionRecall test times:",len(offSetList))
    for oneOffSet in offSetList:
#        print("----", oneOffSet)
        
        pa = [1 * oneOffSet, 0]
        pb = [-1 * oneOffSet, 0]

        df = cpfWarningLevelTest(cpf, warningLevel, pa, pb)
        
        oneValue = {"pa":pa, "pb":pb, "df":df}
        decisionValues.append(oneValue)
        
    simulationValue ={"cpf":cpf,
                      "warningLevel":warningLevel,
                      "decisionValues":decisionValues}
    
    return simulationValue
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    