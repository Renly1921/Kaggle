# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import operator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

def toInt(array):
    array = np.mat(array)
    m,n = np.shape(array)
    newArray = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
                newArray[i,j] = int(array[i,j])
    return newArray

def nomalizing(array):
    m,n = np.shape(array)
    for i in range(m):
        for j in range(n):
            if array[i,j]!=0:
                array[i,j]=1
    return array

def loadTrainData():
    l = []
    with open('train.csv') as file:
         lines = csv.reader(file)
         for line in lines:
             l.append(line) #42001*785
    l.remove(l[0])
    l = np.array(l)
    label = l[:,0]
    data = l[:,1:]
    return nomalizing(toInt(data)),toInt(label)

def loadTestData():
    l = []
    with open('test.csv') as file:
         lines = csv.reader(file)
         for line in lines:
             l.append(line) #28001*784
    l.remove(l[0])
    data = np.array(l)
    return nomalizing(toInt(data))

def classify(inX, dataSet, labels, k):
    inX = np.mat(inX)
    dataSet = np.mat(dataSet)
    labels = np.mat(labels)
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = np.array(diffMat)**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[0,sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def saveResult(result):
    #with open('result.csv') as csvfile:
    #    rows = csv.reader(csvfile)
    #    tmplist = []
     #   for row in rows:
     #       tmplist.append(row)
     #       print(row)
     #   print(tmplist)
    with open('result.csv','w', newline='') as myFile:
        myWriter = csv.writer(myFile)
        for i in result:
            tmp = []
            tmp.append(int(i))
            myWriter.writerow(tmp)

def handwritingClassTest():
    trainData,trainLabel = loadTrainData()
    testData = loadTestData()
    m,n = np.shape(testData)
    resultList=[]
    for i in range(m):
         classifierResult = classify(testData[i], trainData, trainLabel, 5)
         resultList.append(classifierResult)
         print("the classifier came back with: %d" % classifierResult)
    saveResult(resultList)

handwritingClassTest()
