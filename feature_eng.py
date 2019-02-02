# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 16:30:51 2019
project : customer product prediction
@author: chandler qian

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
import re

#  free memory - garbage collector
import gc

gc.collect()

train_set = pd.read_csv(r'.\preprocess\train_set.csv', index_col=0)
train_set.reset_index(inplace=True)
train_set['province_code'] = train_set['province_code'].astype('str')

# %%
# encoding of the train_set
# let's look at the existing columns and pick up the feature needs encoding.
# columns ['date', 'cust_id', 'employee_index', 'country', 'sex', 'age',
#       'open_days', 'last_6_months_flag', 'seniority', 
#       'last_date_primary', 'customer_type', 'customer_relation',
#       'domestic_index', 'foreigner_index', 'channel', 'province_code',
#       'activity_index', 'gross_income', 'segment']

# use one-hot encoding since not enough time for other encoding test
# use one-hot to calculate variance first, 
# label encoding is not working well with variance feature selection, so out of scope for now
cat_fea_label = ['employee_index', 'country', 'sex', 'customer_type', 'customer_relation',
                 'domestic_index', 'foreigner_index', 'channel', 'province_code', 'segment']

non_cat_label = ['age', 'open_days', 'last_6_months_flag', 'seniority', 'last_date_primary',
                 'activity_index', 'gross_income']

pri_key = ['date', 'cust_id']


# use minmax scalr at this time, want to make consisnt with other binary features.

def nor_scaler(fe_list):
    min_max_scaler = preprocessing.MinMaxScaler()

    x_train_mima = min_max_scaler.fit_transform(fe_list)

    return x_train_mima


# figure out seniority has anomaly, so most will be 0.99, remove outlier.
# the income has high range, most income will be around 0 since max is extreme high, 
# no time to change solution at this time.
train_set.groupby(['seniority']).size()
train_set['seniority'].replace({-999999: train_set['seniority'].value_counts().index[0]}
                               , inplace=True)

train_set[['age', 'open_days', 'seniority', 'gross_income']] = nor_scaler(
    train_set[['age', 'open_days', 'seniority', 'gross_income']])

test_set['seniority'].replace({-999999: test_set['seniority'].value_counts().index[0]}
                              , inplace=True)

test_set[['age', 'open_days', 'seniority', 'gross_income']] = nor_scaler(
    test_set[['age', 'open_days', 'seniority', 'gross_income']])

# pandas's powerful onehot encoding method
onehot = pd.get_dummies(train_set[cat_fea_label], prefix=None, prefix_sep='_')
onehot_test = pd.get_dummies(test_set[cat_fea_label], prefix=None, prefix_sep='_')


# another way is to use label encoder which is tested however not prefered in this case
# def label_en(df, column):
#    
#    le = preprocessing.LabelEncoder(sparse=True)
#    le.fit(df[column])
#    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#    df[column].replace(le_name_mapping, inplace=True)

# for column in cat_fea_label:
#    
#    label_en(test, column)


# initial feature selection to reduce features. we can try alot in this step.
# chi2, variance threshold, random forest to select feature importance etc..
def var_filter(df, threshold=0.2):
    filt = VarianceThreshold(threshold)
    filt.fit(df)
    return df[df.columns[filt.get_support(indices=True)]]


# not enough memory from my coumpter, have to chunk into few pieces
# and find the variance for features from high to low
def var_selector(df, upper, lower, step):
    for threshold in np.arange(upper, lower, -step):
        print('current testing threshold ' + str(threshold))
        fea_list = []
        i = 0

        while i < (len(df.columns) // 100 + 1):

            try:
                feature = var_filter(onehot.iloc[:, (i - 1) * 100:(i * 100)], threshold=threshold)
                i += 1
                fea_list.append(feature.columns.values)

            except:
                print('no feature pass the threshold at NO.%s' % (i) + ' chunk')
                i += 1

        if len(fea_list) > 0:
            print(fea_list)

        else:
            print('no feature has higher variance than %s' % (threshold))


# ideal call
var_selector(onehot, 1, 0, 0.01)

# loopthrough the variance and find the optimum point which can be expensive.
# returns information : current testing threshold 0.200000000000000046
# no feature pass the threshold at NO.0 chunk
# no feature pass the threshold at NO.1 chunk
# ['sex_H', 'sex_V', 'customer_relation_A', 'customer_relation_I','channel_KAT'] at NO.2 chunk
# ['channel_KFA', 'channel_KFC', 'channel_KHE', 'province_code_11.0','province_code_15.0'] at NO.3 chunk
# current testing threshold 0.10000000000000005
# no feature pass the threshold at NO.0 chunk
# no feature pass the threshold at NO.1 chunk
# ['sex_H', 'sex_V', 'customer_relation_A', 'customer_relation_I','channel_KAT'] at NO.2 chunk
# ['channel_KFA', 'channel_KFC', 'channel_KHE', 'channel_KHK','province_code_11.0', 'province_code_14.0', 'province_code_15.0',
# 'province_code_18.0'] at NO.3 chunk 

onehot_fea_final = ['sex_H', 'sex_V', 'customer_relation_A', 'customer_relation_I'
    , 'channel_KAT', 'channel_KFA', 'channel_KFC', 'channel_KHE'
    , 'channel_KHK', 'province_code_11.0', 'province_code_14.0'
    , 'province_code_15.0', 'province_code_18.0']

# time to put trainset together
x_train_final = train_set[(pri_key + non_cat_label)].join(onehot[onehot_fea_final])
x_test_final = test_set[(pri_key + non_cat_label)].join(onehot_test[onehot_fea_final])


# correlation coefficient check for features selected and manually picked features.
def plot_cor(fe_df):
    corr = fe_df.corr()
    plt.figure(figsize=(16, 16))
    corr_map = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,
                           annot=True)
    figure = corr_map.get_figure()
    figure.savefig('./vis/corr_conf.png')


# plot and save corr matrix, for sex_H and sex_V has -1 correlation for sure, no worry
# l2 regulization can handle this. for activity_index and cust relation has also high relation
# which is good to know.
plot_cor(x_train_final.iloc[:, 2:])

x_train_final.to_csv(r'.\preprocess\x_train_final.csv')
# 

# %% relabel the train account, we need to find action of customers instead of existing accounts
# we use the changes from previous to target month to build label, so intotal 15 months.
# For example Jan 2015 we have customer  A B C D, Feb we have customer B C E. we only 
# consider customer B C and their accounts as valid label for Feb data. 
# this can be computational very expensive.

# train_acc.to_csv(r'.\preprocess\train_acc.csv')
# train_acc = pd.read_csv(r'.\preprocess\train_acc.csv', index_col=0)
# train_acc['cust_key'] = train_acc['cust_key'].astype('str')

delta = pd.merge(train_acc[train_acc['date'] == date_list[1]]
                 , train_acc[train_acc['date'] == date_list[2]], on=['cust_key'], suffixes=('/last', '/current'))


# get a super large df first with dfs of both last month and current month
def cust_monthly_label(df):
    y_train = pd.DataFrame()
    date_list = df.groupby('date').size().index.tolist()

    for i in range(0, len(date_list) - 1):
        #                   len(date_list)):
        in_join = pd.merge(df[df['date'] == date_list[i]]
                           , df[df['date'] == date_list[i + 1]], on=['cust_key'], suffixes=('/last', '/current'))
        y_train = y_train.append(in_join)

    return y_train


y_train = cust_monthly_label(train_acc)


# get the delta value for each account with cust_key and current date
def delta_table(df):
    y_train_final = pd.DataFrame()
    y_train_final['date'] = df['date/current']
    y_train_final['cust_key'] = df['cust_key']
    al_col = df.columns.tolist()
    regex_la = re.compile(".*/last")
    regex_cu = re.compile(".*/current")
    #   exclude the date_last which is not needed for delta
    last_cols = list(filter(regex_la.match, al_col))[1:]
    print(last_cols)
    #    cu_cols = list(filter(regex_cu.match, al_col))

    for col in last_cols:
        delta_name = col.split('/')[0]
        y_train_final[delta_name] = df[delta_name + '/current'] - df[col]

    return y_train_final


y_train_final = delta_table(y_train)


def fre_des(df):
    for column in df.columns:
        print(df[column].value_counts(dropna=False).nlargest(20))


fre_des(y_train_final)
# -1 value found which is totally right, but we don't need to investigate at this point

y_train_final.replace({-1: 0}, inplace=True)

y_train_final.to_csv(r'.\preprocess\y_train_final.csv')

# validation step
# jan = train_acc[train_acc['date'] == '2015-01-28']['cust_key']
# feb = train_acc[train_acc['date'] == '2015-02-28']['cust_key']
# jan.to_csv(r'.\preprocess\jan_acc.csv')
# feb.to_csv(r'.\preprocess\feb_acc.csv')

# %%
