# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 23:10:04 2017

@author: Lei Su
"""

from numpy import *  
import SVM  
  
################## test svm #####################  
## step 1: load data  
print "step 1: load data..."  
#dataSet = []  
#labels = []  
#fileIn = open('testSet.txt')  
#for line in fileIn.readlines():  
#    lineArr = line.strip().split('\t')  
#    dataSet.append([float(lineArr[0]), float(lineArr[1])])  
#    labels.append(float(lineArr[2]))  
#  
#dataSet = mat(dataSet)  
#labels = mat(labels).T  
#train_x = dataSet[0:60, :]  
#train_y = labels[0:60, :]  
#test_x = dataSet[40:101, :]  
#test_y = labels[40:101, :]  
import numpy as np
import pandas as pd
from pandas import DataFrame
#data normalization
def inputData(X):
    rows,cols = X.shape
    for j in range(cols):
        max = np.max(X[:,j],axis=0)
        min = np.min(X[:,j],axis=0)
        for i in range(rows):
            X[i,j]=(float(X[i,j])-min)/(max-min)
    return X
def labelData(Y):
    for i in range(Y.shape[0]):
        if Y[i][0]<1:
            Y[i][0]=-1
    return Y
DataLoad_bot = pd.read_csv('bots_data.csv')
DataSet_bot = DataLoad_bot[['followers_count','friends_count','listedcount','favourites_count','verified','statuses_count','default_profile','has_extended_profile','bot']]
DataLoad_nonbot = pd.read_csv('nonbots_data.csv')
DataSet_nonbot = DataLoad_nonbot[['followers_count','friends_count','listedcount','favourites_count','verified','statuses_count','default_profile','has_extended_profile','bot']]
DataSet = pd.concat([DataSet_bot,DataSet_nonbot])
DataSet = mat(DataSet,dtype=np.float32)
DataSet = np.random.permutation(DataSet)
startnum_train = 1800
endnum_train = 2197
train_x = inputData(DataSet[:startnum_train,:8])#columns=['followers_count','friends_count','listed_count','favourites_count','verified','statuses_count','default_profile','default_profile_image','has_extended_profile'])
#print (train_x)
#train_x = DataSet[:80,:9]
#print(train_x[1,2])
train_y = labelData(DataSet[:startnum_train,8:9])#,columns=['bot'])
#print (train_y)
test_x = inputData(DataSet[startnum_train:endnum_train,:8])#,columns=['followers_count','friends_count','listed_count','favourites_count','verified','statuses_count','default_profile','default_profile_image','has_extended_profile'])
#print (test_x)
test_y = labelData(DataSet[startnum_train:endnum_train,8:9])#,columns=['bot'])
#print (test_y)
## step 2: training...  
print "step 2: training..."  
C = 0.6  
toler = 0.001
maxIter = 50 
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('linear', 1.0))  
print (svmClassifier.alphas.A)
## step 3: testing  
print "step 3: testing..."  
accuracy = SVM.testSVM(svmClassifier, test_x, test_y)  
  
## step 4: show the result  
print "step 4: show the result..."    
print 'The classify accuracy is: %.3f%%' % (accuracy * 100)  

