# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:48:43 2021

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
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

'-----------------------------------------------------------------------------'
#import the dataset and get the initial statistics
# SKIP THIS BLOCK IF YOU ARE ALREADY IMPORTING A DF (except sensitive attr and decision label definition)
dataset_orig, privileged_groups, unprivileged_groups = aif_data("adult", 1, False)

sens_attr = ['race', 'sex']
decision_label = 'income-per-year'
# Getting the baseline peformance results from the imbalanced dataset
fav_l = 1
unfav_l = 0

# Initial disparities in the full original dataset
metric_orig = BinaryLabelDatasetMetric(dataset_orig, 
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)
print("Disparate impact (of original labels) between unprivileged and privileged groups = %f" % metric_orig.disparate_impact())
print("Difference in statistical parity (of original labels) between unprivileged and privileged groups = %f" % metric_orig.statistical_parity_difference())
print("Individual fairness metric that measures how similar the labels are for similar instances = %f" % metric_orig.consistency())

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
orig_df.reset_index(drop=True, inplace=True)
orig_df['sub_labels'].value_counts()

#OR, EQUALIZE THE POSITIVE AND THE NEGATIVE CLASS SIZES
from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy='majority')
X = orig_df.loc[:, orig_df.columns != 'income-per-year']
y = orig_df.loc[:, orig_df.columns == 'income-per-year']
y=y.astype('int')
X_over, y_over = undersample.fit_resample(X, y)
new_df = pd.concat([X_over, pd.DataFrame(y_over)], axis=1)
orig_df = new_df
print(orig_df['sub_labels'].value_counts())

'----------------------------------------------------------------------------'
# Train-test split WITH stratification
X = orig_df.loc[:, orig_df.columns != decision_label]
y = orig_df.loc[:, orig_df.columns == decision_label].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42, shuffle=True,
                                                    stratify=X['sub_labels'])

# Check class imbalance in the splitted training set
print(X_train['sub_labels'].value_counts())
print(X_test['sub_labels'].value_counts())

# Partial feture scaling (of numerical variables)
X_train, X_test = scale(X_train, X_test)
num_list, cat_list = type_lists(X_train)

#Calculate the base fairness metrics that can be obtained from the original dataset
#Find the privileged and unprivileged subgroups based on the X_train's original labels
dataset_metrics, aggr_metrics, priv_gr, unpriv_gr = aif_dataset_metrics(X_train, y_train, 
                                                                        sens_attr, fav_l, unfav_l)

'----------------------------------------------------------------------------'
# Getting the baseline performance results from the imbalanced dataset
# Note: the function is created based on the assump. that the X's have sub_labels
# Instantiate the desired classifier obj to train the classification models
classifier = LogisticRegression()
# classifier = RandomForestClassifier()
# classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
#                                     max_depth=1, random_state=0)

baseline_stats, cm, ratio_table = baseline_metrics(classifier, X_train, X_test, 
                                                  y_train, y_test, sens_attr, 
                                                  fav_l, unfav_l)
'-----------------------------------------------------------------------------'

# Keep the subgroup labels to append them back later
keep_sub_l = X_train['sub_labels']

# Required drops for the GERMAN dataset (THIS DF CREATION IS A MUST)
X_train_new = X_train.drop(['age', 'sex', 'sub_labels'], axis=1)

# Get the idx of categ and numeric columns again due to the column drops above
num_list, cat_list = type_lists(X_train_new)

'-----------------------------------------------------------------------------'
'''K-PROTOTYPES'''
# Elbow method for kprototypes
# Note: The min and max num of cluster to try must be given as input
elbow_plot = kprot_elbow(2, 10, X_train_new, cat_list)

# Actual clustering with k-prototypes
nc = 4
model = KPrototypes(n_clusters=nc, init='Cao', cat_dissim=matching_dissim)
clusters = model.fit_predict(X_train_new, categorical=cat_list)
silhouette_val = silhouette_score(X_train_new, clusters, metric='manhattan')
print(silhouette_val)

# t-sne plotting for2D visualization (adjust t-SNE hyperparameters in the function def)
plot_tsne(X_train_new,clusters, 60, 100)


'----------------------------------------------------------------------------'
# Putting the required label info back to the dataframe before oversampling
X_train_new['cluster_labels'] = clusters
X_train_new['cluster_labels'] = X_train_new['cluster_labels'].astype('object')
X_train_new['sub_labels'] = keep_sub_l
# Also put the original decision labels so that they are also oversampled
X_train_new['class_labels'] = y_train


#cluster datasets in their original form
existing_clust = {}
for h in range(len(X_train_new['cluster_labels'].unique())):
   existing_clust[h] = X_train_new.loc[X_train_new['cluster_labels']==h]
#checking the subgroup counts in each cluster dataset
for item in existing_clust:
    print(existing_clust[item]['sub_labels'].value_counts())

#fixing the cluster memberships in each df if a sample from a subgroup is alone
fixed_clusters = fix_memberships(X_train_new, model)
for item in fixed_clusters:
    print(fixed_clusters[item]['sub_labels'].value_counts())

#transform the data types of all the columns to numeric for SMOTE
for df in fixed_clusters:
    for i in range(len(fixed_clusters[df].columns)):       
        fixed_clusters[df].iloc[:,i] = fixed_clusters[df].iloc[:,i].astype('float')

# Over-sampling of each cluster
oversampled_clusters, unique_subl = oversample(fixed_clusters)
for item in oversampled_clusters:
    print(oversampled_clusters[item]['sub_labels'].value_counts())
    
# Deleting sensitive attributes and subgroup labels from test set is required
# to apply the implemented solutions (sens. attr. are not used to satisfy the
# disparate treatment in the functions)
test_sublabels = X_test['sub_labels']
X_test_n = X_test.drop(['age', 'sex','sub_labels'], axis=1)
num_list, cat_list = type_lists(X_test_n)

'----------------------------------------------------------------------------'

# Predicting the test sets based on strategy 1
X_test_pred1 = predict_whole_set(classifier, oversampled_clusters, X_test_n)

'----------------------------------------------'
# Predicting the test sets based on strategy 2
costs = labels_cost(X_test_n, model.cluster_centroids_, euclidean_dissim,
                    matching_dissim, 0.5)
X_test_pred2 = predict_per_model(classifier, oversampled_clusters, model, X_test_n, costs,
                                 cat_list, unique_subl, test_sublabels)

'----------------------------------------------'
# Predicting the test sets based on strategy 3
X_test_pred3 = predict_w_weights(classifier, oversampled_clusters, costs, 
                                 X_test_n, unique_subl, test_sublabels)

'----------------------------------------------------------------------------'
'''The metrics table creation for given dataset'''

# Protected attributes and groups must be defined based on the dataset and
# preferences to calculate fairness and performance metrics

metrics_table1, cm1, ratio_t1 = metrics_calculate(X_test, X_test_pred1, y_test, sens_attr,
                                        fav_l, unfav_l)
metrics_table2, cm2, ratio_t2 = metrics_calculate(X_test, X_test_pred2, y_test, sens_attr,
                                        fav_l, unfav_l)
metrics_table3, cm3, ratio_t3 = metrics_calculate(X_test, X_test_pred3, y_test, sens_attr,
                                        fav_l, unfav_l)
#if all reasults are needed to be placed in one dataframe
# all_results = pd.concat([baseline_stats, metrics_table1, metrics_table2,
#                          metrics_table3], axis=0)