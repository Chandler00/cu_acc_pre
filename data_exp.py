# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 21:17:19 2019

@author: chandler qian

"""

import pandas as pd

#   1. data exploration
# load data
train_set = pd.read_csv('./train/train_monthly_info.csv')
train_acc = pd.read_csv('./train/train_accounts.csv')
train_ref = pd.read_csv('./train/xref.csv')
test_set =  pd.read_csv('./test/test_monthly_info.csv')

# basic information
train_set.head(5)
train_acc.head(5)

# raw data type are not currectly read, ideally should be string, numeric, date.
train_set.dtypes

# find only payroll is float, could because of nan value.
train_acc.dtypes
train_acc['payroll'] = train_acc['payroll'].astype('int')

# create frequency describer, see basic information
def fre_des(df):
    
    for column in df.columns:
        print (df[column].value_counts(dropna=False).nlargest(20))

# ' NA' value found for several columns
fre_des(train_set)
fre_des(train_acc)

train_set.describe()
train_acc.describe()

train_set.info()
train_acc.info()

# null value exploration, ' NA' not included
acc_null_ratio = train_acc.isnull().sum(axis=0).sort_values(ascending=False)/len(train_acc)
set_null_ratio = train_set.isnull().sum(axis=0).sort_values(ascending=False)/len(train_set)

# 2. preprocessing
# remove the column which contains very little information
# 'deceased_status' , 'primary_address' - only 1 value, 'province_name' -duplicate with pro code, 
# 'spouse_index' 99.9% missing
train_set.drop(['primary_address', 'province_name', 'spouse_index'], axis =1, inplace=True)
train_set = train_set[train_set['deceased_status']=='N']
train_set.drop(['deceased_status'], axis =1, inplace=True)

# serious preprocessing for this task needed.
# start from convert dataframe columns to proper datatype

# narrow down to a single column 'age' which can't be converted to 'int'. 
# ' NA'found and all rows contains ' NA' are empty. 
na_rows = train_set[train_set['age'].astype(str).str.contains('NA', case=False)]
na_rows.isnull().sum(axis=0).sort_values(ascending=False)/len(na_rows)

# remove empty rows. 27734 rows removed. update null ratio analysis
train_set = train_set[~train_set['age'].astype(str).str.contains('NA', case=False)]
#set_null_ratio = train_set.isnull().sum(axis=0).sort_values(ascending=False)/len(train_set)

# handle null values. listed below are several ways of handing it, it depends the logic.
# 1. fill the null value with the most common value 2. create a new column for the null as a new category
# 3. if numeric replace by mean, median 4. if a column are mostly null, depends on the corrleation. most likely drop will a good solution

# function to replace with most common type
def re_common(df, column):
    
    common = df[column].value_counts().index[0]
    
    df[column].fillna(common, inplace =True)
    


for column in ['segment', 'channel', 'customer_relation', 'customer_type', 
                'province_code', 'sex']:
    
    re_common(train_set, column)

# replace na from gross income to mean or median
train_set['gross_income'].fillna(train_set['gross_income'].mean(skipna=True), inplace=True)

# if last_date_primary is confirm then return 1  if null then 0.
def replace_value(content):
    
    if pd.isnull(content):
        
        pass
    
    else :
        return 1

train_set['last_date_primary'] = train_set['last_date_primary'].map(replace_value)       
train_set['last_date_primary'].fillna(0, inplace=True)

    
#train_set['last_date_primary'].fillna(0, inplace=True)

# consolidate data type

def con_type(df, column, typestr):
    
    df[column] = df[column].astype(typestr)
 
for column in ['age', 'last_6_months_flag', 'seniority', 'primary', 'last_date_primary']:
    
    con_type(train_set, column, 'int')

# replace and simplify values
train_set['customer_type'] = train_set['customer_type'].astype('str')
train_set['customer_type'].replace({'1.0':'pr', '1':'pr', '2':'co', '2.0':'co'
         , '3':'fo_pr', '3.0':'fo_pr', '4.0':'fo_co', '4':'fo_co'}, inplace=True)
    
train_set['domestic_index'].replace({'S':1, 'N':0}, inplace=True)
train_set['foreigner_index'].replace({'S':1, 'N':0}, inplace=True)







