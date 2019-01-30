# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 16:30:51 2019

@author: chandler qian

"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
# to free memory - garbage collector
import gc
gc.collect()

#train_set = pd.read_csv(r'.\preprocess\train_set.csv')
#train_set= train_set.iloc[:, 1:]
#train_set.reset_index(inplace= True)
#train_set['province_code'] = train_set['province_code'].astype('str')

#%%
# encoding of the train_set
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
def var_selector(df):
    
    for threshold in np.arange(0.6, 0, -0.05):
        print('current testing threshold ' + str(threshold))
        fea_list = []
        i=0
        while i < (len(df.columns)//100 +1):
            
            try:
                feature = var_filter(onehot.iloc[:, (i-1)*100:(i*100)], threshold=threshold)
                i+=1
                fea_list.append(feature.columns)
            except:
                print('feature not found with %s'%s (i))
                i+=1
                
        print (fea_list)

# loopthrough the variance and find the optimum point
var_selector(onehot)

for i in np.arange(1, 0,-0.05):
    
    print(i)


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