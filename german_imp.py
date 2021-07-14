# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 13:47:23 2021

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
import skfuzzy as fuzz

#if the user uploads a dataset, this function is not used
data_name = "german"
dataset_orig, privileged_groups, unprivileged_groups = aif_data(data_name, False)

# Define sensitive attributes and decision label names for subroup label function
# Note: Sensitive attribute(s) must be always given as a list
sens_attr = ['age', 'sex']
decision_label = 'credit'
fav_l = 1
unfav_l = 0

#the initial metrics from the original dataset
metric_orig = BinaryLabelDatasetMetric(dataset_orig, 
                                              unprivileged_groups=unprivileged_groups,
                                              privileged_groups=privileged_groups)
di = metric_orig.disparate_impact()
dpd = metric_orig.statistical_parity_difference()
consistency = metric_orig.consistency()


# Creating the snythetic sub-class label column and num-cat columns identification
orig_df, num_list, cat_list = preprocess(dataset_orig, sens_attr, decision_label)


# Train-test split WITH stratification
X = orig_df.loc[:, orig_df.columns != decision_label]
y = orig_df.loc[:, orig_df.columns == decision_label].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, 
                                                    shuffle=True,
                                                    stratify=X['sub_labels'])

# Partial feture scaling (of numerical variables)
X_train, X_test = scale(X_train, X_test)
num_list, cat_list = type_lists(X_train)

clf = RandomForestClassifier()
# NOTE: clf must come from the user!
# Getting the baseline performance results from the imbalanced dataset
# Note: the function is created based on the assumption that the X's have sub_labels
# Instantiate the desired classifier obj to train the classification models    
baseline_stats, cm, ratio_table, preds = baseline_metrics(clf, X_train, X_test, 
                                                  y_train, y_test, sens_attr, 
                                                  fav_l, unfav_l)



# Keep the subgroup labels to append them back later
keep_sub_l = X_train['sub_labels']

# Required drops for the GERMAN dataset (THIS DF CREATION IS A MUST)
X_train_new = X_train.drop(['age', 'sex', 'sub_labels'], axis=1)

# Get the idx of categ and numeric columns again due to the column drops above
num_list, cat_list = type_lists(X_train_new)

# Dimensionality reduction for big datasets with FAMD
X_train_new['sub_labels'] = keep_sub_l

famd = FAMD(n_components=2, random_state = 42)
famd.fit(X_train_new.drop('sub_labels', axis=1))
X_train_reduc = famd.transform(X_train_new)

# Delete the subgroup label column again if dimensionality reduction is used
X_train_new = X_train_new.drop(['sub_labels'], axis=1)
'---------------------------------------------------'
#clustering implementation with fuzzy c-means
alldata = np.vstack((X_train_reduc[0], X_train_reduc[1]))
#FUZZY C-MEANS PLOTTING
#Set up the loop and plot
def fuzzy_2d_plot():
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
    fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
    fpcs = []
    
    #checking for the optimal num of clusters with FPC plots
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
    
    #fpc plot per number of clusters
    fig2, ax2 = plt.subplots()
    ax2.plot(np.r_[2:11], fpcs)
    ax2.set_xlabel("Number of centers")
    ax2.set_ylabel("Fuzzy partition coefficient")
    return fig1,  fig2
'-------------------------------------------------------'
def silhouette_plot():
    from tqdm import tqdm
    n_clusters = []
    silhouette_scores = []
    for i in tqdm(range(2, 10)):
        try:
            cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(alldata, i, 2, error=0.005, 
                                                              maxiter=5000)
            u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(X_train_reduc.T, cntr, 2, error=0.005,       
                                                                maxiter=5000)
            clusters = np.argmax(u, axis=0)
            silhouette_val = silhouette_score(X_train_reduc, clusters, 
                                              metric='euclidean')
            silhouette_scores.append(silhouette_val)
            n_clusters.append(i)
        except:
            print(f"Can't cluster with {i} clusters")
    plt.scatter(x=n_clusters, y=silhouette_scores)
    plt.plot(n_clusters, silhouette_scores)
    plt.show()
'----------------------------------------------------------'
#NOTE: n_clust must be given by the user!
def cluster(n_clust, X_train_reduc):
    cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(alldata, n_clust, 2, error=0.005, 
                                                      maxiter=5000)
    # u: final fuzzy-partitioned matrix, u0: initial guess at fuzzy c-partitioned matrix,
    # d: final euclidean distance matrix, jm: obj func hist, p: num of iter run, 
    #fpc: fuzzy partition coefficient
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(X_train_reduc.T, cntr, 2, error=0.005,       
                                                       maxiter=5000)
    clusters = np.argmax(u, axis=0)  # Hardening for silhouette
    return clusters, cntr
'------------------------------------------------------------'
def process_the_rest(X_train_new, clusters, y_train, keep_sub_l, X_train_reduc, cntr):
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
    
    
    #fixing the cluster memberships in each df if a sample from a subgroup is alone
    fixed_clusters = fix_memberships_fcm(X_train_new, X_train_reduc, clust_centroids=cntr)
    
        
    #transform the data types of all the columns to numeric for SMOTE
    for df in fixed_clusters:
        for i in range(len(fixed_clusters[df].columns)):       
            fixed_clusters[df].iloc[:,i] = fixed_clusters[df].iloc[:,i].astype('float')
    
    # Over-sampling of each cluster
    oversampled_clusters, unique_subl = oversample(fixed_clusters)
    
    # Deleting sensitive attributes and subgroup labels from test set is required
    # to apply the implemented solutions (sens. attr. are not used to satisfy the
    # disparate treatment in the functions)
    test_sublabels = X_test['sub_labels']
    X_test_n = X_test.drop(['age', 'sex','sub_labels'], axis=1)
    num_list, cat_list = type_lists(X_test_n)
    X_test_reduc = famd.transform(X_test_n)
    
'--------------------------------------------------------------'
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
'--------------------------------------------------------------------'

def metrics_strategy1(X_test, X_test_pred1, y_test, sens_attr, fav_l, unfav_l):
    metrics_table1, cm1, ratio_t1 = metrics_calculate(X_test, X_test_pred1, y_test,
                                                  sens_attr, fav_l, unfav_l)
    return metrics_table1, cm1, ratio_t1

def metrics_strategy2(X_test, X_test_pred2, y_test, sens_attr, fav_l, unfav_l):
    metrics_table2, cm2, ratio_t2 = metrics_calculate(X_test, X_test_pred2, y_test,
                                                  sens_attr, fav_l, unfav_l)
    return metrics_table2, cm2, ratio_t2

def metrics_strategy3(X_test, X_test_pred3, y_test, sens_attr, fav_l, unfav_l):
    metrics_table3, cm3, ratio_t3 = metrics_calculate(X_test, X_test_pred3, y_test,
                                                  sens_attr, fav_l, unfav_l)
    return metrics_table3, cm3, ratio_t3