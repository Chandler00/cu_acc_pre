# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 16:49:14 2019
project : customer product prediction
@author: chandler qian
"""
import pandas as pd
import numpy as np
from skmultilearn.problem_transform.br import BinaryRelevance
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
from skmultilearn.adapt import MLkNN
from sklearn.metrics import average_precision_score, hamming_loss
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

#x_train_final = pd.read_csv(r'.\preprocess\x_train_final.csv', index_col=0)
#y_train_final = pd.read_csv(r'.\preprocess\y_train_final.csv', index_col=0)
#x_test_final = pd.read_csv(r'.\preprocess\x_test_final.csv', index_col=0)
# re-indexing is needed rows dropped on x_train, some leftover work of preprocessing

x_train_final['cust_id'] = x_train_final['cust_id'].astype('str')
index = pd.read_csv(r'.\train\xref.csv')
index['cust_id']=index['cust_id'].astype(str)
index['cust_key']=index['cust_key'].astype(str)


x_train_final = pd.merge(x_train_final, index, on = ['cust_id'])
x_train_final.drop(columns =['cust_id'], inplace=True)
x_train_final['cust_key'] = x_train_final['cust_key'].astype('str')
y_train_final['cust_key'] = y_train_final['cust_key'].astype('str')
x_y_train = pd.merge(x_train_final, y_train_final, on = ['date','cust_key'])
x_y_train.reset_index(inplace=True)

#x_y_train.to_csv(r'.\preprocess\x_y_train_final.csv')
#x_y_train = pd.read_csv(r'.\preprocess\x_y_train_final.csv')

# reset x_train, y_train
feature_list = ['age', 'open_days', 'last_6_months_flag', 'seniority',
       'last_date_primary', 'activity_index', 'gross_income', 'sex_H', 'sex_V',
       'customer_relation_A', 'customer_relation_I', 'channel_KAT',
       'channel_KFA', 'channel_KFC', 'channel_KHE', 'channel_KHK',
       'province_code_11.0', 'province_code_14.0', 'province_code_15.0',
       'province_code_18.0']

label_list = ['savings_account', 'guarantees',
       'current_accounts', 'derived_account', 'payroll_account',
       'junior_account', 'more_particular_account', 'particular_account',
       'particular_plus_account', 'short_term_deposits',
       'medium_term_deposits', 'long_term_deposits', 'e_account', 'funds',
       'mortgage', 'pensions', 'loans', 'taxes', 'credit_card', 'securities',
       'home_account', 'payroll', 'direct_debt']

x = x_y_train[feature_list]
y = x_y_train[label_list]

def fre_des(df):
    
    for column in df.columns:
        print (df[column].value_counts(dropna=False).nlargest(20))
fre_des(y)

# giveup some feature not to predict at this time.
# basically savings_account, guarantees, derived_account, junior_account, mortgage
# pensions, loans, home_account, 'securities', 'medium_term_deposits', 'more_particular_account'
# 'particular_account', 'particular_plus_account', 'funds' can be ignored since less than 0.01% population of positive.

y = y[['current_accounts', 'payroll_account',  'short_term_deposits',
        'long_term_deposits', 'e_account','taxes',
       'credit_card', 'payroll', 'direct_debt']]

test = x_test_final[feature_list]
test.to_csv(r'.\preprocess\test.csv')

# split train test set.
x_train, x_test, y_train, y_test = cross_validation.train_test_split(
            x, y, test_size=0.3, random_state=0)

        
# multi-label KNN
# use binary reference method to develop
# kfold is not used in this case since not enough time
# ideally use gridsearch to tune parameters..
# however the MLKNN library does not support ball tree or LSH, so it's very expensive to calculate

parameters = {'k': range(3,8), 's': [0.5, 0.75, 1.0]}
#parameters = {'k': range(3,4), 's': [ 1.0]}
score = 'f1_macro'
clf_knn = GridSearchCV(MLkNN(), parameters, scoring=score)
#clf_knn.fit(x_train.iloc[:1000, :].values, y_train.iloc[:1000,:].values)
clf_knn.fit(x_train.values, y_train.values)
print (clf_knn.best_params_, clf_knn.best_score_)
ave_pre = average_precision_score(y_test, clf_knn.predict(x_test))

# SVM with binary reference method, we can use grid search to tune the parameter. however 
# i don't have enough time to run it.
# PCA first then SVM. PCA sometime used without feature selection. for simplicity we used pca after feature selection
# we suppose need a loop here to try different PCA components. which will take couple hours..
pca = PCA(n_components=5)
pca.fit(x_train)
x_pca = pca.transform(x_train)
x_pca_te = pca.transform(x_test)

# for svm try different kernel and C for regulization strength, class_weight is highly
# this takes probably few hours for my computer to run.

parameters = {'kernel':('rbf', 'poly', 'linear'), 'C':[1, 2,3,4,5, 10]}
clf_svm = OneVsRestClassifier(SVC(class_weight="balanced"))
clf = GridSearchCV(clf_svm, parameters, cv=5)
clf.fit(x_pca, y_train)
clf.score(x_pca_te, y_test)
clf.best_estimator_



