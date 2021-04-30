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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score

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
orig_df.reset_index(drop=True, inplace=True)

#check correlation of the columns
res = orig_df.apply(lambda x : pd.factorize(x)[0] if (x.dtype == 'O') else x).corr(method='pearson', min_periods=1)
#check the correlation of features to class labels
res.loc[res.iloc[:,56] > 0.1, 'important_columns'] = res.iloc[:,56]

#DECREASE THE NUM OF SAMPLES TEMPORARILY WITH STRATIFICATION
i_class0 = orig_df.loc[orig_df['sub_labels']==6]
i_class1 = orig_df.loc[orig_df['sub_labels']==7]
df_rest = orig_df.loc[orig_df['sub_labels']!=6]
n_class0 = len(i_class0)
n_class1 = len(i_class1)
df_class_0_under = i_class0.sample(n_class1)
df_under = pd.concat([df_class_0_under, df_rest], axis=0)

i_class0 = df_under.loc[orig_df['sub_labels']==2]
i_class1 = df_under.loc[orig_df['sub_labels']==3]
df_rest = df_under.loc[orig_df['sub_labels']!=2]
n_class0 = len(i_class0)
n_class1 = len(i_class1)
df_class_0_under = i_class0.sample(n_class1)
df_final = pd.concat([df_class_0_under, df_rest], axis=0)
print(df_final['sub_labels'].value_counts())
orig_df = df_final

'----------------------------------------------------------------'
# Train-test split WITH stratification
X = orig_df.loc[:, orig_df.columns != decision_label]
y = orig_df.loc[:, orig_df.columns == decision_label].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42,
                                                    stratify=X['sub_labels'])


# Train-test split WITHOUT stratification
X = orig_df.loc[:, orig_df.columns != decision_label]
y = orig_df.loc[:, orig_df.columns == decision_label].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Check class imbalance in the splitted training set
print(X_train['sub_labels'].value_counts())
print(X_test['sub_labels'].value_counts())

# Partial feture scaling (of numerical variables)
X_train, X_test = scale(X_train, X_test)
num_list, cat_list = type_lists(X_train)

'---------------------------------------------------------------------------'
# Getting the baseline peformance results from the imbalanced dataset
baseline_stats, cm = baseline_metrics(X_train, X_test, y_train, y_test,
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

famd = FAMD(n_components = 2, n_iter = 3, random_state = 42)
famd.fit(X_train_new.drop('sub_labels', axis=1))
X_train_reduc = famd.transform(X_train_new)
ax = famd.plot_row_coordinates(X_train_new, 
                               color_labels=['sub-labels {}'.format(t) for t in X_train_new['sub_labels']] )
famd.explained_inertia_
X_train_red = famd.partial_row_coordinates(X_train_new)
ax = famd.plot_partial_row_coordinates(X_train_new, 
                                       color_labels=['sub-labels {}'.format(t) for t in X_train_new['sub_labels']])
# Delete the subgroup label column again if dimensionality reduction is used
X_train_new = X_train_new.drop(['sub_labels'], axis=1)

#if importing gower's distance from csv is needed
gd = pd.read_csv('')

'----------------------------------------------------------------------------'
# Elbow method for kprototypes
# Note: The min and max num of cluster to try must be given as input
elbow_plot = kprot_elbow(5, 15, X_train_new, cat_list)

'''K-PROTOTYPES'''
# Actual clustering with k-prototypes
nc = 9
model = KPrototypes(n_clusters=nc, init='Cao')
clusters = model.fit_predict(X_train_new, categorical=cat_list)
silhouette_val = silhouette_score(X_train_new, clusters, metric='cosine')
print(silhouette_val)


'''FUZZY C-MEANS'''
import skfuzzy as fuzz
alldata = np.vstack((X_train_reduc[0], X_train_reduc[1]))

# Set up the loop and plot
colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
fpcs = []
#checking for th optimal num of clusters
for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

    # Store fpc values for later
    fpcs.append(fpc)

    # Plot assigned clusters, for each data point in training set
    cluster_membership = np.argmax(u, axis=0)
    for j in range(ncenters):
        ax.plot(X_train_reduc[0][cluster_membership == j],
                X_train_reduc[1][cluster_membership == j], '.', color=colors[j])

    # Mark the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1], 'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, fpc))
    ax.axis('off')
fig1.tight_layout()

#fpc visualization per num of clusters (1 best 0 worst)
fig2, ax2 = plt.subplots()
ax2.plot(np.r_[2:11], fpcs)
ax2.set_xlabel("Number of centers")
ax2.set_ylabel("Fuzzy partition coefficient")

#predict with the num of clusters desired
cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(alldata, 6, 2, error=0.005, 
                                                  maxiter=5000)
# u: final fuzzy-partitioned matrix, u0: initial guess at fuzzy c-partitioned matrix,
# d: final euclidean distance matrix, jm: obj func hist, p: num of iter run, 
#fpc: fuzzy partition coefficient
u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(X_train_reduc.T, cntr, 2, error=0.005,       
                                                   maxiter=5000)
clusters = np.argmax(u, axis=0)  # Hardening for silhouette   
print(silhouette_score(X_train_reduc, clusters, metric='manhattan'))



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

#fixing cluster memberships for FUZZY C-MEANS
fixed_clusters = fix_memberships_fcm(X_train_new, X_train_reduc, clust_centroids=cntr)
#checking the subgroup counts in each cluster dataset
for item in fixed_clusters:
    print(fixed_clusters[item]['sub_labels'].value_counts())
    
    
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

'----------------------------------------------------------------------------'
# Predicting the test sets based on strategy 1
X_test_pred1 = predict_whole_set(oversampled_clusters, X_test_n)

'-----------------------------------------'

# Predicting the test sets based on strategy 2
costs = labels_cost(X_test_n, model.cluster_centroids_, euclidean_dissim,
                    matching_dissim, 0.5)
X_test_pred2 = predict_per_model(oversampled_clusters, model, X_test_n, costs,
                                 cat_list, unique_subl, test_sublabels)

#IF IT IS A FUZZY C-MEANS MODEL:
X_test_reduc = famd.transform(X_test_n)
#t_u is needed later for option 3
t_u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    X_test_reduc.T, cntr, 2, error=0.005, maxiter=1000)
test_clusts = np.argmax(t_u, axis=0) 
X_test_pred2 = predict_w_fuzzy(oversampled_clusters, X_test_n, X_test_reduc, 
                                  cntr, unique_subl, test_sublabels) 


'-------------------------------------------'
# Predicting the test sets based on strategy 3
'''K-PROTOTYPES'''
X_test_pred3 = predict_w_weights(oversampled_clusters, costs, X_test_n,
                                 unique_subl, test_sublabels)


'''FUZZY C-MEANS'''
X_test_pred3 = predict_w_weights_fuzzy(oversampled_clusters, t_u, X_test_n,
                                       unique_subl, test_sublabels)


'----------------------------------------------------------------------------'
'''The metrics table creation for given dataset'''

# Protected attributes and groups must be defined based on the dataset and
# preferences to calculate fairness and performance metrics

metrics_table1, cm1 = metrics_calculate(X_test, X_test_pred1, y_test, sens_attr)
metrics_table2, cm2 = metrics_calculate(X_test, X_test_pred2, y_test, sens_attr)
metrics_table3, cm3 = metrics_calculate(X_test, X_test_pred3, y_test, sens_attr)

all_results = pd.concat([baseline_stats, metrics_table1, metrics_table2,
                         metrics_table3], axis=0)