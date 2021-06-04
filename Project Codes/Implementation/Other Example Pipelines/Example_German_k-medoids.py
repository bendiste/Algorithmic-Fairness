# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:41:40 2021

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
import matplotlib.cm as cm

import matplotlib.pyplot as plt

'-----------------------------------------------------------------------------'
#import the dataset and get initial statistics

# SKIP THIS BLOCK IF YOU ARE ALREADY IMPORTING A DATAFRAME FROM A CSV(except sensitive attr and decision label definition)
dataset_orig, privileged_groups, unprivileged_groups = aif_data("german", 2, False)


# Define sensitive attributes and decision label names for subroup label function
# Note: Sensitive attribute(s) must be always given as a list
sens_attr = ['age', 'sex']
decision_label = 'credit'
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

# The list of sub-group sizes in the dataset (to monitor the dist. of sub-groups)
orig_df['sub_labels'].value_counts()

#check correlation of the columns
# res = orig_df.apply(lambda x : pd.factorize(x)[0] if (x.dtype == 'O') else x).corr(method='pearson', min_periods=1)
#check the correlation of features to class labels
# res.loc[res.iloc[:,58].abs() > 0.25, 'important_columns'] = res.iloc[:,58]
#plot heatmap
# plt.figure(figsize=(16,12))
# _ = sns.heatmap(res)

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

baseline_stats, cm = baseline_metrics(classifier, X_train, X_test, 
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
'''POSSIBLE PRE-PROCESSING OPTIONS'''
# Gower dist to get distance matrix (for k-medoids)
import gower
cat = [True if X_train_new[x].dtype == 'object' else False for x in X_train_new.columns]
gd = gower.gower_matrix(X_train_new, cat_features = cat)

#OR (can be used for both fuzzy c-means and k-medoids):
# Optional dimensionality reduction for big datasets with FAMD
X_train_new['sub_labels'] = keep_sub_l
famd = FAMD(n_components=2, random_state = 42)
famd.fit(X_train_new.drop('sub_labels', axis=1))
X_train_reduc = famd.transform(X_train_new)
#plotting the reduced dimensions
# ax = famd.plot_row_coordinates(X_train_new, 
#                                 color_labels=['sub-labels {}'.format(t) for t in X_train_new['sub_labels']] )
# X_train_red = famd.partial_row_coordinates(X_train_new)
# famd.explained_inertia_
# ax = famd.plot_partial_row_coordinates(X_train_new, 
#                                         color_labels=['sub-labels {}'.format(t) for t in X_train_new['sub_labels']])
# Delete the subgroup label column again if dimensionality reduction is used
X_train_new = X_train_new.drop(['sub_labels'], axis=1)

'----------------------------------------------------------------------------'

'''K-MEDOIDS'''
from sklearn_extra.cluster import KMedoids

#Note: when the metric is precomputed, object doesnt return any cluster centers.
#find the num of cluster based on inertia
#if gower's distance is used, then the metric params should be 'precomputed'
costs = []
n_clusters = []
clusters_assigned = []
silhouette_scores = []
from tqdm import tqdm
for i in tqdm(range(2, 10)):
    try:
        cluster = KMedoids(n_clusters=i, metric='euclidean', method='pam',
                    random_state=0).fit(X_train_reduc)
        clusters = cluster.predict(X_train_reduc)
        costs.append(cluster.inertia_)
        n_clusters.append(i)
        clusters_assigned.append(clusters)
        silhouette_val = silhouette_score(X_train_reduc, clusters, 
                                          metric='euclidean')
        silhouette_scores.append(silhouette_val)
    except:
        print(f"Can't cluster with {i} clusters")
       
plt.scatter(x=n_clusters, y=costs)
plt.plot(n_clusters, costs)
plt.show()

plt.scatter(x=n_clusters, y=silhouette_scores)
plt.plot(n_clusters, silhouette_scores)
plt.show()

#predict cluster labels (3 or 6 clusts)
numc = 3
model = KMedoids(n_clusters=numc, metric='euclidean', method='pam',
                    random_state=0).fit(X_train_reduc)
clusters = model.predict(X_train_reduc)
centroids_df = X_train_new.iloc[model.medoid_indices_]
cluster_centroids = centroids_df.reset_index(drop=True)
cluster_centroids = cluster_centroids.to_numpy(np.float64)
if len(model.cluster_centers_) == 0:
    model.cluster_centroids_ = cluster_centroids
    model.cluster_centers_ = cluster_centroids
else:
    pass

'--------------------------------------------------------'

#if gower's distance is used
plot_tsne(gd, clusters, 60, 80)
#if FAMD is used
plot_tsne(X_train_reduc, clusters, 50, 100)

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
#if gower's distance is used:
# test_set = pd.concat([centroids_df, X_test_n], axis=0)
# cats = [True if X_test_n[x].dtype == 'object' else False for x in X_test_n.columns]
# gd_test = gower.gower_matrix(test_set, cat_features = cats)
# pairwise_dist = gd_test[numc:, 0:numc]

#if FAMD is used:
X_test_reduc = famd.transform(X_test_n)
dists = kmed_dists(X_test_reduc, model.cluster_centers_)

X_test_pred2 = predict_w_clusters(classifier, oversampled_clusters, X_test_n, 
                                  dists, unique_subl, test_sublabels) 

'----------------------------------------------'
# Predicting the test sets based on strategy 3
'''K-PROTOTYPES'''
X_test_pred3 = predict_w_weights_kmed(classifier, oversampled_clusters, dists, 
                                      X_test_n, unique_subl, test_sublabels)

'----------------------------------------------------------------------------'
'''The metrics table creation for given dataset'''

# Protected attributes and groups must be defined based on the dataset and
# preferences to calculate fairness and performance metrics

metrics_table1, cm1 = metrics_calculate(X_test, X_test_pred1, y_test, sens_attr,
                                        fav_l, unfav_l)
metrics_table2, cm2 = metrics_calculate(X_test, X_test_pred2, y_test, sens_attr,
                                        fav_l, unfav_l)
metrics_table3, cm3 = metrics_calculate(X_test, X_test_pred3, y_test, sens_attr,
                                        fav_l, unfav_l)
#if all reasults are needed to be placed in one dataframe
all_results = pd.concat([baseline_stats, metrics_table1, metrics_table2,
                         metrics_table3], axis=0)