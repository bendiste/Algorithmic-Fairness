# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:42:06 2021

@author: hatta
"""

from implementation_functions import *

import pandas as pd
import numpy as np
from prince import FAMD #Factor analysis of mixed data
from aif360.metrics import BinaryLabelDatasetMetric
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from pandas import ExcelWriter

#COMPAS IMPLEMENTATION
data_name = "compas"
dataset_orig, privileged_groups, unprivileged_groups = aif_data(data_name, False)
#assign the sensitive attr and decision labels
sens_attr = ['race', 'sex']
decision_label = 'two_year_recid'
fav_l = 1
unfav_l = 0

# Initial disparities in the full original dataset
metric_orig = BinaryLabelDatasetMetric(dataset_orig, 
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)
print("Disparate impact (of original labels) between unprivileged and privileged groups = %f" % metric_orig.disparate_impact())
print("Difference in statistical parity (of original labels) between unprivileged and privileged groups = %f" % metric_orig.statistical_parity_difference())
print("Individual fairness metric that measures how similar the labels are for similar instances = %f" % metric_orig.consistency())

# in this dataset, 'protected_attribute_maps': [sex={0.0: 'Male', 1.0: 'Female'}, 
# race={1.0: 'Caucasian', 0.0: 'Not Caucasian'}]
# class label maps': [{1.0: 'Did recid.', 0.0: 'No recid.'}]
orig_df, num_list, cat_list = preprocess(dataset_orig, sens_attr, decision_label)

#switch the dataset labels other way around for easier interpretation
orig_df['transf_labels'] = np.where(orig_df['two_year_recid']== 0, 1, 0)
decision_label = 'transf_labels'
orig_df = orig_df.drop('two_year_recid', axis=1)

orig_df, num_list, cat_list = preprocess(orig_df, sens_attr, decision_label)
orig_df['sub_labels'].value_counts()

#clean the extra columns
cols = [c for c in orig_df.columns if (c.lower()[:13] != 'c_charge_desc')]
orig_df = orig_df[cols]
#or statement did not work so I do it twice
cols2 = [c for c in orig_df.columns if (c.lower()[:7] != 'age_cat')]
orig_df = orig_df[cols2]

# Train-test split WITH stratification
X = orig_df.loc[:, orig_df.columns != decision_label]
y = orig_df.loc[:, orig_df.columns == decision_label].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    shuffle=True,
                                                    stratify=X['sub_labels'])
#for extra insight regarding the dataset (because of stratification, both must be almost identical)
#full_ratio_table = subgroup_ratios(orig_df, sens_attr)
#base_ratio_table = subgroup_ratios(X_test, sens_attr)

# Check class imbalance in the splitted training set
print(X_train['sub_labels'].value_counts())
print(X_test['sub_labels'].value_counts())

# Partial feture scaling (of numerical variables)
X_train, X_test = scale(X_train, X_test)
num_list, cat_list = type_lists(X_train)

clf = RandomForestClassifier()

#Calculate the base fairness metrics that can be obtained from the original dataset
#Find the privileged and unprivileged subgroups based on the X_train's original labels
dataset_metrics, aggr_metrics, priv_gr, unpriv_gr = aif_dataset_metrics(X_train, y_train, 
                                                                  sens_attr, fav_l, unfav_l)

'----------------------------------------------------------------------------'

# Type the desired classifier to train the classification models with classifier obj        
baseline_stats, cm, ratio_table = baseline_metrics(clf, X_train, X_test, y_train, 
                                      y_test, sens_attr, fav_l, unfav_l)

'----------------------------------------------------------------------------'

# Keep the subgroup labels to append them back later
keep_sub_l = X_train['sub_labels']

# Required drops for the GERMAN dataset (THIS DF CREATION IS A MUST)
X_train_new = X_train.drop(['race', 'sex', 'sub_labels'], axis=1)

# Get the idx of categ and numeric columns again due to the column drops above
num_list, cat_list = type_lists(X_train_new)

'-----------------------------------------------------------------------------'
# Dimensionality reduction for big datasets with FAMD (for fuzzly c-means)

famd = FAMD(n_components=2, random_state = 42)
famd.fit(X_train_new)
X_train_reduc = famd.transform(X_train_new)
# ax = famd.plot_row_coordinates(X_train_new, 
#                                 color_labels=['sub-labels {}'.format(t) for t in X_train_new['sub_labels']] )
# X_train_red = famd.partial_row_coordinates(X_train_new)
# famd.explained_inertia_
# ax = famd.plot_partial_row_coordinates(X_train_new, 
#                                         color_labels=['sub-labels {}'.format(t) for t in X_train_new['sub_labels']])
# Delete the subgroup label column again if dimensionality reduction is used

'----------------------------------------------------------------------------'
#clustering implementation with fuzzy c-means
alldata = np.vstack((X_train_reduc[0], X_train_reduc[1]))

# Set up the loop and plot
# colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
# fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
# fpcs = []
# # checking for the optimal num of clusters
# for ncenters, ax in enumerate(axes1.reshape(-1), 2):
#     cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
#         alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

#     # Store fpc values for later
#     fpcs.append(fpc)

#     # Plot assigned clusters, for each data point in training set
#     cluster_membership = np.argmax(u, axis=0)
#     for j in range(ncenters):
#         ax.plot(X_train_reduc[0][cluster_membership == j],
#                 X_train_reduc[1][cluster_membership == j], '.', color=colors[j])

#     # Mark the center of each fuzzy cluster
#     for pt in cntr:
#         ax.plot(pt[0], pt[1], 'rs')

#     ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
#     ax.axis('off')
# fig1.tight_layout()

# #fpc visualization per num of clusters
# fig2, ax2 = plt.subplots()
# ax2.plot(np.r_[2:11], fpcs)
# ax2.set_xlabel("Number of centers")
# ax2.set_ylabel("Fuzzy partition coefficient")
# #clear the existing plot for the next silhouette plotting
# fig1.clear()
# fig2.clear()
# plt.close(fig1)
# plt.close(fig2)

# #silhouette score plot per number of cluster
# from tqdm import tqdm
# n_clusters = []
# silhouette_scores = []
# for i in tqdm(range(2, 10)):
#     try:
#         cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(alldata, i, 2, error=0.005, 
#                                                           maxiter=5000)
#         u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(X_train_reduc.T, cntr, 2, error=0.005,       
#                                                             maxiter=5000)
#         clusters = np.argmax(u, axis=0)
#         silhouette_val = silhouette_score(X_train_reduc, clusters, 
#                                           metric='euclidean')
#         silhouette_scores.append(silhouette_val)
#         n_clusters.append(i)
#     except:
#         print(f"Can't cluster with {i} clusters")
# plt.scatter(x=n_clusters, y=silhouette_scores)
# plt.plot(n_clusters, silhouette_scores)
# plt.show()


#either 4,5 or 6 clsuters
#predict with the num of clusters desired
cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(alldata, 5, 2, error=0.005, 
                                                  maxiter=5000)
# u: final fuzzy-partitioned matrix, u0: initial guess at fuzzy c-partitioned matrix,
# d: final euclidean distance matrix, jm: obj func hist, p: num of iter run, 
#fpc: fuzzy partition coefficient
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(X_train_reduc.T, cntr, 2, error=0.005,       
                                                   maxiter=5000)
clusters = np.argmax(u, axis=0)  # Hardening for silhouette   
print(silhouette_score(X_train_reduc, clusters, metric='cosine'))


'''VISUALIZATION IN 2D'''
# t-sne plotting for2D visualization
# plot_tsne(X_train_reduc, clusters, 60, 100)

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
fixed_clusters = fix_memberships_fcm(X_train_new, X_train_reduc, clust_centroids=cntr)
#checking the subgroup counts in each cluster dataset
for item in fixed_clusters:
    print(fixed_clusters[item]['sub_labels'].value_counts())


#transform the data types of all the columns to numeric for SMOTE
for df in fixed_clusters:
    for i in range(len(fixed_clusters[df].columns)):       
        fixed_clusters[df].iloc[:,i] = fixed_clusters[df].iloc[:,i].astype('float')
    # print(fixed_clusters[df].dtypes)

# Over-sampling of each cluster
oversampled_clusters, unique_subl = oversample(fixed_clusters)
for item in oversampled_clusters:
    print(oversampled_clusters[item]['sub_labels'].value_counts())
    
# Deleting sensitive attributes and subgroup labels from test set is required
# to apply the implemented solutions (sens. attr. are not used to satisfy the
# disparate treatment in the functions)
test_sublabels = X_test['sub_labels']
X_test_n = X_test.drop(['race', 'sex','sub_labels'], axis=1)
num_list, cat_list = type_lists(X_test_n)
X_test_reduc = famd.transform(X_test_n)

'----------------------------------------------------------------------------'
# Predicting the test sets based on strategy 1
X_test_pred1 = predict_whole_set(clf, oversampled_clusters, X_test_n)

# Predicting the test sets based on strategy 2
#t_u is needed later for option 3
t_u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    X_test_reduc.T, cntr, 2, error=0.005, maxiter=1000)
test_clusts = np.argmax(t_u, axis=0) 
X_test_pred2 = predict_w_fuzzy(clf, oversampled_clusters, X_test_n, 
                               X_test_reduc, cntr, unique_subl, test_sublabels) 

# Predicting the test sets based on strategy 3
X_test_pred3 = predict_w_weights_fuzzy(clf, oversampled_clusters, t_u, 
                                       X_test_n, unique_subl, test_sublabels)


'----------------------------------------------------------------------------'
'''The metrics table creation for given dataset'''

# Protected attributes and groups must be defined based on the dataset and
# preferences to calculate fairness and performance metrics

metrics_table1, cm1, ratio_t1 = metrics_calculate(X_test, X_test_pred1, 
                                                  y_test, sens_attr,
                                                  fav_l, unfav_l)
metrics_table2, cm2, ratio_t2 = metrics_calculate(X_test, X_test_pred2,
                                                  y_test, sens_attr,
                                                  fav_l, unfav_l)
metrics_table3, cm3, ratio_t3 = metrics_calculate(X_test, X_test_pred3,
                                                  y_test, sens_attr,
                                                  fav_l, unfav_l)