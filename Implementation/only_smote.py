# -*- coding: utf-8 -*-
"""
Created on Mon May 10 09:57:37 2021

@author: hatta
"""

#ONLY SMOTE IMPLEMENTATION
from imblearn.over_sampling import SMOTE
from implementation_functions import *

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# any dataset that previously uploaded
x = X_train_new
y = keep_sub_l.astype('float')

sm = SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=1)

x_res, y_res = sm.fit_resample(x, y)
new_df = pd.concat([x_res, pd.DataFrame(y_res)], axis=1)
print(new_df['sub_labels'].value_counts())

#prepare the test set (don't forget to change the sens. attr. per dataset)
X_test_n = X_test.drop(['race', 'sex','sub_labels'], axis=1)
num_list, cat_list = type_lists(X_test_n)

#train and predict in the way of option 1
lr = LogisticRegression()
X_train = new_df.loc[:, new_df.columns != 'class_labels']
X_train = X_train.loc[:,X_train.columns != 'sub_labels']
y_train = new_df.loc[:, new_df.columns == 'class_labels']
y_train = np.asarray(y_train).flatten()
y_train = y_train.astype('float')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test_n)
X_test_pred = pd.DataFrame(X_test.copy())
X_test_pred['y_pred'] = y_pred
X_test_pred['race'] = X_test_pred['race'].astype('float')
X_test_pred['sex'] = X_test_pred['sex'].astype('float')

#get the metrics table
metrics_table_sm, cm1 = metrics_calculate(X_test, X_test_pred, y_test, sens_attr,
                                        fav_l, unfav_l, priv_gr, unpriv_gr)
