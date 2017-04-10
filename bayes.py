# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 23:55:10 2017

@author: Lei Su
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

DataLoad_bot = pd.read_csv('bots_data.csv')
DataSet_bot = DataLoad_bot[['followers_count','friends_count','listedcount','favourites_count','verified','statuses_count','default_profile','has_extended_profile','bot']]
DataLoad_nonbot = pd.read_csv('nonbots_data.csv')
DataSet_nonbot = DataLoad_nonbot[['followers_count','friends_count','listedcount','favourites_count','verified','statuses_count','default_profile','has_extended_profile','bot']]
DataSet = pd.concat([DataSet_bot,DataSet_nonbot])
DataSet = np.random.permutation(DataSet)
startnum_train = 1800
endnum_train = 2197
train_data = DataFrame(DataSet[:startnum_train,:9],columns=['followers_count','friends_count','listed_count','favourites_count','verified','statuses_count','default_profile','has_extended_profile','bot'])
train_label = DataFrame(DataSet[:startnum_train,8:9],columns=['bot'])
test_data = DataFrame(DataSet[startnum_train:endnum_train,:8],columns=['followers_count','friends_count','listed_count','favourites_count','verified','statuses_count','default_profile','has_extended_profile'])
test_label = DataFrame(DataSet[startnum_train:endnum_train,8:9],columns=['bot'])

#train
p1 = float(train_label.apply(lambda x:x.sum()))/train_label.shape[0]
p0 = 1-p1
#followers_count attribute
range_followers = 100
p1_followers_0 = float(train_data.loc[(train_data['followers_count']<=range_followers)&(train_data['bot']==1)].shape[0]+1)/(train_data.loc[train_data['bot']==1].shape[0]+1)
p1_followers_1 = float(train_data.loc[(train_data['followers_count']>range_followers)&(train_data['bot']==1)].shape[0]+1)/(train_data.loc[train_data['bot']==1].shape[0]+1)
p0_followers_0 = float(train_data.loc[(train_data['followers_count']<=range_followers)&(train_data['bot']==0)].shape[0]+1)/(train_data.loc[train_data['bot']==0].shape[0]+1)
p0_followers_1 = float(train_data.loc[(train_data['followers_count']>range_followers)&(train_data['bot']==0)].shape[0]+1)/(train_data.loc[train_data['bot']==0].shape[0]+1)
#friends_count attribute
range_friends = 100
p1_friends_0 = float(train_data.loc[(train_data['friends_count']<=range_friends)&(train_data['bot']==1)].shape[0]+1)/(train_data.loc[train_data['bot']==1].shape[0]+1)
p1_friends_1 = float(train_data.loc[(train_data['friends_count']>range_friends)&(train_data['bot']==1)].shape[0]+1)/(train_data.loc[train_data['bot']==1].shape[0]+1)
p0_friends_0 = float(train_data.loc[(train_data['friends_count']<=range_friends)&(train_data['bot']==0)].shape[0]+1)/(train_data.loc[train_data['bot']==0].shape[0]+1) 
p0_friends_1 = float(train_data.loc[(train_data['friends_count']>range_friends)&(train_data['bot']==0)].shape[0]+1)/(train_data.loc[train_data['bot']==0].shape[0]+1)
#listed_count attribute
range_listed = 100
p1_listed_0 = float(train_data.loc[(train_data['listed_count']<=range_listed)&(train_data['bot']==1)].shape[0]+1)/(train_data.loc[train_data['bot']==1].shape[0]+1)
p1_listed_1 = float(train_data.loc[(train_data['listed_count']>range_listed)&(train_data['bot']==1)].shape[0]+1)/(train_data.loc[train_data['bot']==1].shape[0]+1)
p0_listed_0 = float(train_data.loc[(train_data['listed_count']<=range_listed)&(train_data['bot']==0)].shape[0]+1)/(train_data.loc[train_data['bot']==0].shape[0]+1)
p0_listed_1 = float(train_data.loc[(train_data['listed_count']>range_listed)&(train_data['bot']==0)].shape[0]+1)/(train_data.loc[train_data['bot']==0].shape[0]+1)
range_favourites = 100
p1_favourites_0 = float(train_data.loc[(train_data['favourites_count']<=range_favourites)&(train_data['bot']==1)+1].shape[0])/(train_data.loc[train_data['bot']==1].shape[0]+1)
p1_favourites_1 = float(train_data.loc[(train_data['favourites_count']>range_favourites)&(train_data['bot']==1)+1].shape[0])/(train_data.loc[train_data['bot']==1].shape[0]+1)
p0_favourites_0 = float(train_data.loc[(train_data['favourites_count']<=range_favourites)&(train_data['bot']==0)+1].shape[0])/(train_data.loc[train_data['bot']==0].shape[0]+1)
p0_favourites_1 = float(train_data.loc[(train_data['favourites_count']>range_favourites)&(train_data['bot']==0)+1].shape[0])/(train_data.loc[train_data['bot']==0].shape[0]+1)
#verified attribute
p1_verified_0 = float(train_data.loc[(train_data['verified']==1)&(train_data['bot']==1)].shape[0]+1)/(train_data.loc[train_data['bot']==1].shape[0]+1)
p1_verified_1 = float(train_data.loc[(train_data['verified']==0)&(train_data['bot']==1)].shape[0]+1)/(train_data.loc[train_data['bot']==1].shape[0]+1)
p0_verified_0 = float(train_data.loc[(train_data['verified']==1)&(train_data['bot']==0)].shape[0]+1)/(train_data.loc[train_data['bot']==0].shape[0]+1)
p0_verified_1 = float(train_data.loc[(train_data['verified']==0)&(train_data['bot']==0)].shape[0]+1)/(train_data.loc[train_data['bot']==0].shape[0]+1)
#statuses_count attribute
range_statuses = 100
p1_statuses_0 = float(train_data.loc[(train_data['statuses_count']<=range_statuses)&(train_data['bot']==1)].shape[0]+1)/(train_data.loc[train_data['bot']==1].shape[0]+1)
p1_statuses_1 = float(train_data.loc[(train_data['statuses_count']>range_statuses)&(train_data['bot']==1)].shape[0]+1)/(train_data.loc[train_data['bot']==1].shape[0]+1)
p0_statuses_0 = float(train_data.loc[(train_data['statuses_count']<=range_statuses)&(train_data['bot']==0)].shape[0]+1)/(train_data.loc[train_data['bot']==0].shape[0]+1)
p0_statuses_1 = float(train_data.loc[(train_data['statuses_count']>range_statuses)&(train_data['bot']==0)].shape[0]+1)/(train_data.loc[train_data['bot']==0].shape[0]+1)
#default_profile attribute
p1_profile_0 = float(train_data.loc[(train_data['default_profile']==1)&(train_data['bot']==1)].shape[0]+1)/(train_data.loc[train_data['bot']==1].shape[0]+1)
p1_profile_1 = float(train_data.loc[(train_data['default_profile']==0)&(train_data['bot']==1)].shape[0]+1)/(train_data.loc[train_data['bot']==1].shape[0]+1)
p0_profile_0 = float(train_data.loc[(train_data['default_profile']==1)&(train_data['bot']==0)].shape[0]+1)/(train_data.loc[train_data['bot']==0].shape[0]+1)
p0_profile_1 = float(train_data.loc[(train_data['default_profile']==0)&(train_data['bot']==0)].shape[0]+1)/(train_data.loc[train_data['bot']==0].shape[0]+1)
#has_extended_profile attribute
p1_extendedprofile_0 = float(train_data.loc[(train_data['has_extended_profile']==1)&(train_data['bot']==1)].shape[0]+1)/(train_data.loc[train_data['bot']==1].shape[0]+1)
p1_extendedprofile_1 = float(train_data.loc[(train_data['has_extended_profile']==0)&(train_data['bot']==1)].shape[0]+1)/(train_data.loc[train_data['bot']==1].shape[0]+1)
p0_extendedprofile_0 = float(train_data.loc[(train_data['has_extended_profile']==1)&(train_data['bot']==0)].shape[0]+1)/(train_data.loc[train_data['bot']==0].shape[0]+1)
p0_extendedprofile_1 = float(train_data.loc[(train_data['has_extended_profile']==0)&(train_data['bot']==0)].shape[0]+1)/(train_data.loc[train_data['bot']==0].shape[0]+1)


#classify
test_bak1 = DataFrame(test_data)
#followers_count
for a0 in test_bak1.loc[test_bak1['followers_count']<=range_followers].index:
    test_bak1['followers_count'].loc[a0] = p1_followers_0
for a1 in test_bak1.loc[test_bak1['followers_count']>range_followers].index:
    test_bak1['followers_count'].loc[a1] = p1_followers_1
#friends_count
for a0 in test_bak1.loc[test_bak1['friends_count']<=range_friends].index:
    test_bak1['friends_count'].loc[a0] = p1_friends_0
for a1 in test_bak1.loc[test_bak1['friends_count']>range_friends].index:
    test_bak1['friends_count'].loc[a1] = p1_friends_1
#listed_count
for a0 in test_bak1.loc[test_bak1['listed_count']<=range_listed].index:
    test_bak1['listed_count'].loc[a0] = p1_listed_0
for a1 in test_bak1.loc[test_bak1['listed_count']>range_listed].index:
    test_bak1['listed_count'].loc[a1] = p1_listed_1
#favourites_count
for a0 in test_bak1.loc[test_bak1['favourites_count']<=range_favourites].index:
    test_bak1['favourites_count'].loc[a0] = p1_favourites_0
for a1 in test_bak1.loc[test_bak1['favourites_count']>range_favourites].index:
    test_bak1['favourites_count'].loc[a1] = p1_favourites_1
#verified
for a0 in test_bak1.loc[test_bak1['verified']==1].index:
    test_bak1['verified'].loc[a0] = p1_verified_0
for a1 in test_bak1.loc[test_bak1['verified']==0].index:
    test_bak1['verified'].loc[a1] = p1_verified_1
#status
for a0 in test_bak1.loc[test_bak1['statuses_count']<=range_statuses].index:
    test_bak1['statuses_count'].loc[a0] = p1_statuses_0
for a1 in test_bak1.loc[test_bak1['statuses_count']>range_statuses].index:
    test_bak1['statuses_count'].loc[a1] = p1_statuses_1
#default_profile
for a0 in test_bak1.loc[test_bak1['default_profile']==1].index:
    test_bak1['default_profile'].loc[a0] = p1_profile_0
for a1 in test_bak1.loc[test_bak1['default_profile']==0].index:
    test_bak1['default_profile'].loc[a1] = p1_profile_1
#has_extended_profile
for a0 in test_bak1.loc[test_bak1['has_extended_profile']==1].index:
    test_bak1['has_extended_profile'].loc[a0] = p1_extendedprofile_0
for a1 in test_bak1.loc[test_bak1['has_extended_profile']==0].index:
    test_bak1['has_extended_profile'].loc[a1] = p1_extendedprofile_1
test_bak1['prob'] = test_bak1['followers_count']*test_bak1['friends_count']*test_bak1['listed_count']*test_bak1['favourites_count']*test_bak1['verified']*test_bak1['statuses_count']*test_bak1['default_profile']*test_bak1['has_extended_profile']*p1

test_bak0 = DataFrame(test_data)
#followers_count
for a0 in test_bak0.loc[test_bak0['followers_count']<=range_followers].index:
    test_bak0['followers_count'].loc[a0] = p0_followers_0
for a1 in test_bak0.loc[test_bak0['followers_count']>range_followers].index:
    test_bak0['followers_count'].loc[a1] = p0_followers_1
#friends_count
for a0 in test_bak0.loc[test_bak0['friends_count']<=range_friends].index:
    test_bak0['friends_count'].loc[a0] = p0_friends_0
for a1 in test_bak0.loc[test_bak0['friends_count']>range_friends].index:
    test_bak0['friends_count'].loc[a1] = p0_friends_1
#listed_count
for a0 in test_bak0.loc[test_bak0['listed_count']<=range_listed].index:
    test_bak0['listed_count'].loc[a0] = p0_listed_0
for a1 in test_bak0.loc[test_bak0['listed_count']>range_listed].index:
    test_bak0['listed_count'].loc[a1] = p0_listed_1
#favourites_count
for a0 in test_bak0.loc[test_bak0['favourites_count']<=range_favourites].index:
    test_bak0['favourites_count'].loc[a0] = p0_favourites_0
for a1 in test_bak0.loc[test_bak0['favourites_count']>range_favourites].index:
    test_bak0['favourites_count'].loc[a1] = p0_favourites_1
#verified
for a0 in test_bak0.loc[test_bak0['verified']==1].index:
    test_bak0['verified'].loc[a0] = p0_verified_0
for a1 in test_bak0.loc[test_bak0['verified']==0].index:
    test_bak0['verified'].loc[a1] = p0_verified_1
#status
for a0 in test_bak0.loc[test_bak0['statuses_count']<=range_statuses].index:
    test_bak0['statuses_count'].loc[a0] = p0_statuses_0
for a1 in test_bak0.loc[test_bak0['statuses_count']>range_statuses].index:
    test_bak0['statuses_count'].loc[a1] = p0_statuses_1
#default_profile
for a0 in test_bak0.loc[test_bak0['default_profile']==1].index:
    test_bak0['default_profile'].loc[a0] = p0_profile_0
for a1 in test_bak0.loc[test_bak0['default_profile']==0].index:
    test_bak0['default_profile'].loc[a1] = p0_profile_1
#has_extended_profile
for a0 in test_bak0.loc[test_bak0['has_extended_profile']==1].index:
    test_bak0['has_extended_profile'].loc[a0] = p0_extendedprofile_0
for a1 in test_bak0.loc[test_bak0['has_extended_profile']==0].index:
    test_bak0['has_extended_profile'].loc[a1] = p0_extendedprofile_1
test_bak0['prob'] = test_bak0['followers_count']*test_bak0['friends_count']*test_bak0['listed_count']*test_bak0['favourites_count']*test_bak0['verified']*test_bak0['statuses_count']*test_bak0['default_profile']*test_bak0['has_extended_profile']*p0
result = list(test_bak1['prob']>test_bak0['prob'])

# result_pos = list(test_bak1['prob']>test_bak0['prob'])
# result_neg = list(test_bak1['prob']<test_bak0['prob'])

pred = DataFrame(result,columns=['bot'])

# pred_pos = DataFrame(result_pos, columns=['bot'])
# pred_neg = DataFrame(result_neg, columns=['bot'])
# test_pos = DataFrame(list(test_label['bot']==1), columns=['bot'])
# test_neg = DataFrame(list(test_label['bot']==0), columns=['bot'])
# print(test_label)
# print(test_pos)
# print(test_neg)
# print(pred_pos)
# print(pred_neg)
# print(pred)
# #precision = TP/(TP+FP)
# precision = float(pred_pos.loc[pred_pos['bot']==test_pos['bot']].shape[0])/pred_pos.shape[0]
# print('Precision:', precision)
# #recall = TP/(TP+FN)
# recall = float(pred_pos.loc[pred_pos['bot']==test_pos['bot']].shape[0])/test_pos.shape[0]
# print('Recall:', recall)
# # error_rate = (FN+FP)/N
# # error_rate = 1 - (TP+TN)/N
# error_rate = 1 - float(pred_pos.loc[pred_pos['bot']==test_pos['bot']].shape[0] + pred_neg.loc[pred_neg['bot']==test_neg['bot']].shape[0])/test_data.shape[0]
# print('Error rate:', error_rate)

accuracy = float(pred.loc[pred['bot']==test_label['bot']].shape[0])/pred.shape[0]
print('Predict Accuracy:',accuracy)













