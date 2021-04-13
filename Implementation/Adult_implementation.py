# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 13:05:21 2021

@author: hatta
"""
from implementation_functions import *

import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose

import pandas as pd
import numpy as np
from prince import FAMD #Factor analysis of mixed data
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.model_selection import train_test_split

'----------------------------------------------------------------------------'
#import the dataset and get the initial statistics
# SKIP THIS BLOCK IF YOU ARE ALREADY IMPORTING A DF (except sensitive attr and decision label definition)
dataset_orig, privileged_groups, unprivileged_groups = aif_data("adult", 1, False)

sens_attr = ['race', 'sex']
decision_label = 'income-per-year'

# Initial disparities in the original dataset
metric_orig = BinaryLabelDatasetMetric(dataset_orig, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

print("Disparate impact (of original labels) between unprivileged and privileged groups = %f" % metric_orig.disparate_impact())
print("Difference in statistical parity (of original labels) between unprivileged and privileged groups = %f" % metric_orig.statistical_parity_difference())
print("Individual fairness metric from Zemel et.al. that measures how similar the labels are for similar instances = %f" % metric_orig.consistency())

'----------------------------------------------------------------------------'
# Creating the snythetic sub-class label column and num-cat columns identification
orig_df, num_list, cat_list = preprocess(dataset_orig, sens_attr, decision_label)

#remove unwanted columns in the analysis
cols = [c for c in orig_df.columns if c.lower()[:14] != 'native-country']
orig_df = orig_df[cols]
orig_df = orig_df.drop(['education-num'], axis=1)

#renew the list of categ and numeric columns if you do the processing above
num_list, cat_list = type_lists(orig_df)

# The list of sub-group sizes in the dataset (to monitor the dist. of sub-groups)
orig_df['sub_labels'].value_counts()

#check correlation of the columns
res = orig_df.apply(lambda x : pd.factorize(x)[0] if (x.dtype == 'O') else x).corr(method='pearson', min_periods=1)
#check the correlation of features to class labels
res.loc[res.iloc[:,56] > 0.1, 'important_columns'] = res.iloc[:,56]

'----------------------------------------------------------------'
# Train-test split WITH stratification
X = orig_df.loc[:, orig_df.columns != decision_label]
y = orig_df.loc[:, orig_df.columns == decision_label].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42,
                                                    stratify=X['sub_labels'])

# Check class imbalance in the splitted training set
print(X_train['sub_labels'].value_counts())
print(X_test['sub_labels'].value_counts())

# Partial feture scaling (of numerical variables)
X_train, X_test = scale(X_train, X_test)
num_list, cat_list = type_lists(X_train)

'---------------------------------------------------------------------------'
# Getting the baseline peformance results from the imbalanced dataset
privileged_groups = [{'sex': 1, 'race': 1}]
unprivileged_groups = [{'sex': 0, 'race': 0}]

baseline_stats, cm = baseline_metrics(X_train, X_test, y_train, y_test, 
                                      privileged_groups, unprivileged_groups, 
                                      sens_attr)

'----------------------------------------------------------------------------'
# Keep the subgroup labels to append them back later
keep_sub_l = X_train['sub_labels']
# Required drops for the ADULT dataset (THIS DF CREATION IS A MUST)
X_train_new = X_train.drop(['race', 'sex', 'sub_labels'], axis=1)

# Get the idx of categ and numeric columns again due to the column drops above
num_list, cat_list = type_lists(X_train_new)

'----------------------------------------------------------------------------'
# Optional dimensionality reduction for big datasets with FAMD
X_train_new['sub_labels'] = keep_sub_l

famd = FAMD(n_components =3, n_iter = 3, random_state = 42)
famd.fit(X_train_new.drop('sub_labels', axis=1))
X_train_reduc = famd.transform(X_train_new)
ax = famd.plot_row_coordinates(X_train_new, 
                               color_labels=['sub-labels {}'.format(t) for t in X_train_new['sub_labels']] )
X_train_red = famd.partial_row_coordinates(X_train_new)
famd.explained_inertia_
ax = famd.plot_partial_row_coordinates(X_train_new, 
                                       color_labels=['sub-labels {}'.format(t) for t in X_train_new['sub_labels']])
# Delete the subgroup label column again if dimensionality reduction is used
X_train_new = X_train_new.drop(['sub_labels'], axis=1)

'----------------------------------------------------------------------------'
# Elbow method for kprototypes
# Note: The min and max num of cluster to try must be given as input
elbow_plot = kprot_elbow(5, 15, X_train_new, cat_list)

# Actual clustering with k-prototypes (9 for german, 11 for adult)
nc = 8
kprot = KPrototypes(n_clusters=nc, init='Cao')
clusters = kprot.fit_predict(X_train_new, categorical=cat_list)

#silhouette scores function will be called when implemented

'----------------------------------------------------------------------------'
# Putting the required label info back to the dataframe before oversampling
X_train_new['cluster_labels'] = clusters
X_train_new['cluster_labels'] = X_train_new['cluster_labels'].astype('object')
X_train_new['sub_labels'] = keep_sub_l
# Also put the original decision labels so that they are also oversampled
X_train_new['class_labels'] = y_train


# Over-sampling of each cluster
fixed_clusters = fix_memberships(X_train_new, kprot)
#checking the subgroup counts in each cluster dataset
for item in fixed_clusters:
    print(fixed_clusters[item]['sub_labels'].value_counts())

oversampled_clusters = oversample(fixed_clusters)
for item in oversampled_clusters:
    print(oversampled_clusters[item]['sub_labels'].value_counts())

# Deleting sensitive attributes and subgroup labels from test set is required
# to apply the implemented solutions (sens. attr. are not used to satisfy the
# disparate treatment in the functions)
X_test_n = X_test.drop(['race', 'sex','sub_labels'], axis=1)


'----------------------------------------------------------------------------'
# Predicting the test sets based on strategy 1
X_test_pred1 = predict_whole_set(oversampled_clusters, X_test_n)


# Predicting the test sets based on strategy 2
X_test_pred2 = predict_per_model(oversampled_clusters, kprot, X_test_n, cat_list)


# Predicting the test sets based on strategy 3
costs = labels_cost(X_test_n, kprot.cluster_centroids_, euclidean_dissim,
                    matching_dissim, 0.5)
X_test_pred3 = predict_w_weights(oversampled_clusters, costs, X_test_n)


'----------------------------------------------------------------------------'
'''The metrics table creation for given dataset'''

# Protected attributes and groups must be defined based on the dataset and
# preferences to calculate fairness and performance metrics
privileged_groups = [{'sex': 1, 'race': 1}]
unprivileged_groups = [{'sex': 0, 'race': 0}]


metrics_table1, cm1 = metrics_calculate(X_test, X_test_pred1, y_test, sens_attr, 
                                  privileged_groups, unprivileged_groups)
metrics_table2, cm2 = metrics_calculate(X_test, X_test_pred2, y_test, sens_attr, 
                                  privileged_groups, unprivileged_groups)
metrics_table3, cm3 = metrics_calculate(X_test, X_test_pred3, y_test, sens_attr, 
                                  privileged_groups, unprivileged_groups)

all_results = pd.concat([baseline_stats, metrics_table1, metrics_table2,
                         metrics_table3], axis=0)