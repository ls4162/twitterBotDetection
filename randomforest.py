# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 10:23:46 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 21:25:45 2017

@author: Administrator
"""

import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

def inputData(X):
    rows,cols = X.shape
    for j in range(cols):
        max = np.max(X[:,j],axis=0)
        min = np.min(X[:,j],axis=0)
        for i in range(rows):
            X[i,j]=(float(X[i,j])-min)/(max-min)
    return X


DataLoad = pd.read_csv('training_data_2_csv_UTF.csv')
DataSet0 = DataLoad[['followers_count','friends_count','listedcount','favourites_count','verified','statuses_count','default_profile','has_extended_profile','bot']]
#DataLoad_nonbot = pd.read_csv('nonbots_data.csv')
#DataSet_nonbot = DataLoad_nonbot[['followers_count','friends_count','listedcount','favourites_count','verified','statuses_count','default_profile','has_extended_profile','bot']]
#DataSet = pd.concat([DataSet_bot,DataSet_nonbot])
#DataSet = np.mat(DataSet,dtype=np.float32)
#TestLoad = pd.read_csv('test_data_4_students.csv')
#TestLoad = TestLoad[:575]
#TestSet0 = TestLoad[['description','followers_count','friends_count','listed_count','favorites_count','verified','statuses_count','default_profile','has_extended_profile']]

TestSet0 = pd.read_csv('TestSet.csv')
DataSet1 = DataSet0.fillna(DataSet0.mean())
TestSet1 = TestSet0.fillna(TestSet0.mean())
#print (type(DataSet1.iloc[1,1]))
#TestSet.to_csv('TestSet.csv',index = False)
DataSet = np.array(DataSet1,dtype=np.float64)
TestSet = np.array(TestSet1,dtype=np.float64)
TestSet = inputData(TestSet)
result = []
for i in range(12):
    DataSet = np.random.permutation(DataSet)
    data = inputData(DataSet[:,:8])#columns=['followers_count','friends_count','listed_count','favourites_count','verified','statuses_count','default_profile','default_profile_image','has_extended_profile'])
    #print (train_x)
    #train_x = DataSet[:80,:9]
    #print(train_x[1,2])
    target = DataSet[:,8:9]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, target, test_size=0.01, random_state=0)
    
    clf = RandomForestClassifier(100).fit(X_train, y_train)
    print(clf.score(X_test,y_test))
    pred = clf.predict(TestSet)
    result.append(pred)
pred = pd.DataFrame(result)
s = list(pred.mean()>0.5)
print(len(s))
plt.scatter(TestSet0['followers_count'],TestSet0['friends_count'])

#bot_judge = pd.read_csv('bot_judge.csv')
#print(bot_judge.iloc[573,0])
#temp1 =[]
#for i in range(len(s)):
#    if ((s[i]+bot_judge.iloc[i,0])>0):
#        temp1.append(1)
#    else:
#        temp1.append(0)
#
#result_rforest = pd.DataFrame(temp1,columns=['bot'])
##result_rforest['second']=bot_judge
##result_rforest['rforest']=((result_rforest['first'].bool()) or (result['second'].bool()))
#result_rforest.to_csv('result_rforest.csv',index=False)
##
