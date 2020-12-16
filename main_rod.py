# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 13:35:17 2020

@author: Rod
"""

#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import random
import pandas as pd
from pandas_profiling import ProfileReport
#%%
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics  
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
# from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
# from sklearn.decomposition import PCA 
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
#%%
df1 = pd.read_csv('purchase_detail.csv')
df2 = pd.read_csv('login.csv')
df3 = pd.read_csv('user_info.csv')
df4 = pd.read_csv('user_label_train.csv')
df5 = pd.read_csv('submission.csv')
#%%
ss1 = set(df1['userid'].tolist())
ss2 = set(df2['userid'].tolist())
ss = set(df4['userid'].tolist() + df5['userid'].tolist())
#%%
df1_1 = df1.copy()
df1_t = pd.DataFrame(df1_1.groupby(["userid",'category_encoded'])['total_amount'].agg('sum')).reset_index()
df1_o = pd.DataFrame(df1_1.groupby(["userid",'category_encoded'])['order_count'].agg('sum')).reset_index()
df1_ot = pd.merge(df1_t,df1_o,how='inner',on=['userid','category_encoded'])
df_tt = df1_ot.groupby(['userid','category_encoded']).sum().unstack()
df_tt.fillna(0,inplace=True)
df_tt.columns
col = ['total_amount_{}'.format(str(n).zfill(2)) for n in range(1, 24)] + ['order_count_{}'.format(str(n).zfill(2)) for n in range(1, 24)]
df_tt.columns = col
#%%
df2_1 = df2.copy()
df2_1  = pd.DataFrame(df2_1.groupby("userid")['login_times'].agg('sum')).reset_index()

#%%
df = pd.merge(df_tt,df2_1,how='inner',on='userid')
df = pd.merge(df,df3,how='inner',on='userid')
df_train = pd.merge(df,df4,how='inner',on='userid')
df_train = df_train.drop('birth_year', axis=1)
df_train = df_train.drop('enroll_time', axis=1)
# df_train = df_train.drop('userid', axis=1)
df_train = df_train.dropna(subset=['gender'])
df_train.info()
# df = df.loc[df['userid'].isin(set(df4['userid'].tolist()))]

df_corr = df_train.corr()


#%% EDA 
profile = ProfileReport(df_train, title="Pandas Profiling Report",pool_size=8)
profile
profile.to_file("your_report.html")
#%% Inbalanced the data and split to training, testing


X, y = df_train.iloc[:,:-1],df_train.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
# sm = SMOTE(random_state=2)
# X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
#%%
param = {
    'max_depth':10,  # the maximum depth of each tree
    'eta': 0.1 , # LR
    'objective': 'binary:logistic', 
    # "scale_pos_weight": 1,
    "eval_metric": "auc"
    }  
num_round = 1000 # the number of training iterations
# for early stopping
watchlist = [(dtrain,'train'),(dtest,'test')]
bst = xgb.train(param, dtrain, num_round, evals=watchlist, early_stopping_rounds=10)
preds = bst.predict(dtest)

for i in range(len(preds)):
    if preds[i] > 0.5:
        preds[i] = 1 
    else:
        preds[i] = 0 
# extracting most confident predictions
best_preds = preds

print ('classification_report\n', metrics.classification_report(y_test, best_preds, digits=6))  
print ('confusion_matrix\n', metrics.confusion_matrix(y_test, best_preds))  
print ('accuracy\t', metrics.accuracy_score(y_test, best_preds)) 
#%%
df_sub = pd.merge(df,df5,how='inner',on='userid')
df_sub = df_sub.drop('birth_year', axis=1)
df_sub = df_sub.drop('enroll_time', axis=1)
df_sub = df_sub.dropna(subset=['gender'])
# df_sub = df_sub.drop('userid', axis=1)



#%%

dtrain = xgb.DMatrix(X, label=y)

param = {
    'max_depth':15,  # the maximum depth of each tree
    'eta': 0.1 , # LR
    'objective': 'binary:logistic', 
    # "scale_pos_weight": 1,
    "eval_metric": "auc"
    }  
num_round = 150 # the number of training iterations
# for early stopping
# watchlist = [(dtrain,'train')]
bst = xgb.train(param, dtrain, num_round)

dtest = xgb.DMatrix(df_sub)
preds = bst.predict(dtest)
for i in range(len(preds)):
    if preds[i] > 0.5:
        preds[i] = 1 
    else:
        preds[i] = 0 
# extracting most confident predictions
best_preds = preds


df_sub['label'] = best_preds

df_tmp = df_sub[['userid', 'label']]
df5
df5_1 = df5.merge(df_tmp, how='outer')
df5_1.fillna(1,inplace=True)
df5_1.to_csv('Submission.csv',index=False)

X, y = df_train.iloc[:,:-1],df_train.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
# sm = SMOTE(random_state=2)
# X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
pXaram = {
    'max_depth':10,  # the maximum depth of each tree
    'eta': 1,  # LR
    'objective': 'binary:logistic', 
    "scale_pos_weight": 1,
    "eval_metric": "auc"
#     num_boosting_rounds
    }  

parameters = {
    'max_depth': [5, 10],
    'learning_rate': [0.1,0.5,1],
#     'min_child_weight': [0, 2, 5, 10, 20],
#     'max_delta_step': [0, 0.2, 0.6, 1, 2],
#     'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
}
xlf = xgb.XGBClassifier(max_depth=10, num_round = 200,
                        learning_rate=0.01,
                        n_estimators=20,
#                         silent=True,
                        objective='binary:logistic',
                        nthread=-1,
                        gamma=0,
                        min_child_weight=1,
                        max_delta_step=0,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        colsample_bylevel=1,
                        reg_alpha=0,
                        reg_lambda=1,
                        scale_pos_weight=1,
                        seed=1440,
                        missing=None)

gsearch = GridSearchCV(xlf, param_grid=parameters, scoring='roc_auc', cv=5)
gsearch.fit(X_train, y_train)

print("Best score: %0.3f" % gsearch.best_score_)
best_parameters = gsearch.best_estimator_.get_params()
print("Best parameters set:"%best_parameters)



