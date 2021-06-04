# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:02:47 2021

@author: hatta
"""


'-------------------------------------------------------------'
#original adult dataset with no one-hot encoding
adult_full = pd.read_csv('adult-all.csv',  header=None, na_values='?') 
adult_full = adult_full.rename(columns = {0:'age', 1:'workclass', 2:'remove', 3:'education',
                             4:'education-num', 5:'marital-status', 6:'occupation',
                             7: 'relationship', 8:'race', 9:'sex', 10:'capital-gain',
                             11:'capital-loss', 12:'hours-per-week', 13:'native-country',
                             14:'income-per-year'})
adult_full = adult_full.drop(['remove'], axis=1)
adult_full = adult_full.drop(['education-num'], axis=1)
adult_full.loc[adult_full['income-per-year'] == '>50K', 'income-per-year'] = 1
adult_full.loc[adult_full['income-per-year'] == '<=50K', 'income-per-year'] = 0
adult_full = adult_full.dropna()

adult_full.loc[adult_full['race'] == 'White', 'race'] = 1
adult_full.loc[adult_full['race'] != 1, 'race'] = 0
adult_full.loc[adult_full['sex'] == 'Male', 'sex'] = 1
adult_full.loc[adult_full['sex'] == 'Female', 'sex'] = 0
adult_full = adult_full.reset_index()
adult_full = adult_full.drop(['index'], axis=1)
adult_full, num_list, cat_list = preprocess(adult_full, sens_attr, decision_label)


# The list of sub-group sizes in the dataset (to monitor the dist. of sub-groups)
adult_full['sub_labels'].value_counts()
