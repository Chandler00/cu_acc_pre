# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 16:30:51 2019

@author: chandler qian

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
#  free memory - garbage collector
import gc
gc.collect()

#train_set = pd.read_csv(r'.\preprocess\train_set.csv')
#train_set= train_set.iloc[:, 1:]
#train_set.reset_index(inplace= True)
#train_set['province_code'] = train_set['province_code'].astype('str')

#%%
# encoding of the train_set
# let's look at the existing columns and pick up the feature needs encoding.
# columns ['date', 'cust_id', 'employee_index', 'country', 'sex', 'age',
#       'open_date', 'last_6_months_flag', 'seniority', 
#       'last_date_primary', 'customer_type', 'customer_relation',
#       'domestic_index', 'foreigner_index', 'channel', 'province_code',
#       'activity_index', 'gross_income', 'segment']

# use one-hot encoding since not enough time for other encoding test
# use one-hot to calculate variance first, 
# label encoding is not working well with variance feature selection, so out of scope for now
cat_fea_label = ['country', 'sex', 'customer_type', 'customer_relation',
                 'channel', 'province_code','segment']

# set a copy of existing one for development purpose only
test = train_set[cat_fea_label]

# pandas's powerful onehot encoding method
onehot = pd.get_dummies(test, prefix=None, prefix_sep= '_')

# another way is to use label encoder which is tested however not prefered in this case
#def label_en(df, column):
#    
#    le = preprocessing.LabelEncoder(sparse=True)
#    le.fit(df[column])
#    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#    df[column].replace(le_name_mapping, inplace=True)

#for column in cat_fea_label:
#    
#    label_en(test, column)


# initial feature selection to reduce features. we can try alot in this step.
# chi2, variance threshold, random forest to select feature importance etc..
def var_filter(df, threshold = 0.2):
    
    filt= VarianceThreshold(threshold)
    filt.fit(df)
    return df[df.columns[filt.get_support(indices=True)]]

# not enough memory from my coumpter, have to chunk into few pieces 
# and find the variance for features from high to low
def var_selector(df, upper, lower, step):
    
    for threshold in np.arange(upper, lower, -step):
        print('current testing threshold ' + str(threshold))
        fea_list = []
        i=0
        
        while i < (len(df.columns)//100 +1):
            
            try:
                feature = var_filter(onehot.iloc[:, (i-1)*100:(i*100)], threshold=threshold)
                i+=1
                fea_list.append(feature.columns.values)
                
            except:
                print('no feature pass the threshold at NO.%s'%(i) + ' chunk')
                i+=1
                
        if len(fea_list)>0:
            print (fea_list)
            
        else:
            print ('no feature has higher variance than %s'%(threshold))


# loopthrough the variance and find the optimum point which can be expensive.
# returns information : current testing threshold 0.200000000000000046
#no feature pass the threshold at NO.0 chunk
#no feature pass the threshold at NO.1 chunk
#['sex_H', 'sex_V', 'customer_relation_A', 'customer_relation_I','channel_KAT'] at NO.2 chunk
#['channel_KFA', 'channel_KFC', 'channel_KHE', 'province_code_11.0','province_code_15.0'] at NO.3 chunk
#current testing threshold 0.10000000000000005
#no feature pass the threshold at NO.0 chunk
#no feature pass the threshold at NO.1 chunk
#['sex_H', 'sex_V', 'customer_relation_A', 'customer_relation_I','channel_KAT'] at NO.2 chunk
#['channel_KFA', 'channel_KFC', 'channel_KHE', 'channel_KHK','province_code_11.0', 'province_code_14.0', 'province_code_15.0',
# 'province_code_18.0'] at NO.3 chunk 

var_selector(onehot, 1, 0, 0.05)

# correlation coefficient check for features selected and manually picked features.
def plot_cor(fe_df):
    
    corr = fe_df.corr()
    plt.figure(figsize=(16, 16))
    corr_map = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, 
                           annot=True)
    figure = corr_map.get_figure()
    figure.savefig('./vis/corr_conf.png')

plot_cor(onehot[['sex_V', 'customer_relation_A']])
#
selector = VarianceThreshold(threshold=0.5)
selector.fit_transform(test)

train_set[cat_fea_label]

from pandas import ExcelWriter

writer = ExcelWriter(r".\preprocess\test.xlsx")

test.iloc[0:300000, :].to_excel(writer)

writer.save()

#%%


jan = train_set[train_set['date'] == '2015-01-28']

feb = train_set[train_set['date'] == '2015-02-28']

inv = [i for i in jan['cust_id'] if i not in feb['cust_id']]


# we are developping a model to see the delta of products by month. 
# the current existing products of customers can not be used as label.