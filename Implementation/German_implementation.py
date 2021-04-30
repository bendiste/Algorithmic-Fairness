# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:32:24 2021

@author: Begum Hattatoglu
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
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

import matplotlib.pyplot as plt
import seaborn as sns

'-----------------------------------------------------------------------------'
#import the dataset and get initial statistics

# SKIP THIS BLOCK IF YOU ARE ALREADY IMPORTING A DATAFRAME (except sensitive attr and decision label definition)
dataset_orig, privileged_groups, unprivileged_groups = aif_data("german", 2, False)


# Define sensitive attributes and decision label names for subroup label function
# Note: Sensitive attribute(s) must be always given as a list
sens_attr = ['age', 'sex']
decision_label = 'credit'

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

# The list of sub-group sizes in the dataset (to monitor the dist. of sub-groups)
orig_df['sub_labels'].value_counts()

#check correlation of the columns
res = orig_df.apply(lambda x : pd.factorize(x)[0] if (x.dtype == 'O') else x).corr(method='pearson', min_periods=1)
#check the correlation of features to class labels
res.loc[res.iloc[:,58].abs() > 0.25, 'important_columns'] = res.iloc[:,58]
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

'----------------------------------------------------------------------------'
# Getting the baseline performance results from the imbalanced dataset
# Note: the function is created based on the assump. that the X's have sub_labels

baseline_stats, cm = baseline_metrics(X_train, X_test, y_train, y_test,
                                      sens_attr)

'-----------------------------------------------------------------------------'
# Keep the subgroup labels to append them back later
keep_sub_l = X_train['sub_labels']

# Required drops for the GERMAN dataset (THIS DF CREATION IS A MUST)
X_train_new = X_train.drop(['age', 'sex', 'sub_labels'], axis=1)

# Get the idx of categ and numeric columns again due to the column drops above
num_list, cat_list = type_lists(X_train_new)

'-----------------------------------------------------------------------------'
# Optional dimensionality reduction for big datasets with FAMD (for fuzzly c-means)
X_train_new['sub_labels'] = keep_sub_l

famd = FAMD(n_components=2, random_state = 42)
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
'''K-PROTOTYPES'''
# Elbow method for kprototypes
# Note: The min and max num of cluster to try must be given as input
elbow_plot = kprot_elbow(3, 10, X_train_new, cat_list)

# Actual clustering with k-prototypes (9 for german, 11 for adult)
nc = 6
model = KPrototypes(n_clusters=nc, init='Cao', cat_dissim=matching_dissim)
clusters = model.fit_predict(X_train_new, categorical=cat_list)
silhouette_val = silhouette_score(X_train_new, clusters, metric='cosine')
print(silhouette_val)

'-----------------------------------------------------------------------'
'''POSSIBLE PRE-PROCESSING OPTIONS'''
# Gower dist to get distance matrix (for k-medoids)
import gower
cat = [True if X_train_new[x].dtype == 'object' else False for x in X_train_new.columns]
gd = gower.gower_matrix(X_train_new, cat_features = cat)

# Applying Kernel PCA (foz fuzzy c-means)
from sklearn.decomposition import KernelPCA
from sklearn.cluster import AgglomerativeClustering
kpca = KernelPCA(n_components = 2, kernel = 'cosine', fit_inverse_transform = False)
X = kpca.fit_transform(X_train_new)

'------------------------------------------------------------------------'

'''K-MEDOIDS'''
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_samples, silhouette_score
#Note: when the metric is precomputed, object doesnt return any cluster centers.
#find the num of cluster based on inertia
costs = []
n_clusters = []
clusters_assigned = []
silhouette_scores = []
from tqdm import tqdm
for i in tqdm(range(2, 10)):
    try:
        cluster = KMedoids(n_clusters=i, metric='precomputed', method='pam',
                    random_state=0).fit(gd)
        clusters = cluster.predict(gd)
        costs.append(cluster.inertia_)
        n_clusters.append(i)
        clusters_assigned.append(clusters)
        silhouette_val = silhouette_score(gd, clusters, metric='precomputed')
        silhouette_scores.append(silhouette_val)
    except:
        print(f"Can't cluster with {i} clusters")
       
plt.scatter(x=n_clusters, y=costs)
plt.plot(n_clusters, costs)
plt.show()

plt.scatter(x=n_clusters, y=silhouette_scores)
plt.plot(n_clusters, silhouette_scores)
plt.show()

#predict cluster labels (6 germ or 7 adult)
numc = 7
model = KMedoids(n_clusters=numc, metric='precomputed', method='pam',
                    random_state=0).fit(gd)
clusters = model.predict(gd)
centroids_df = X_train_new.iloc[model.medoid_indices_]
cluster_centroids = centroids_df.reset_index(drop=True)
cluster_centroids = cluster_centroids.to_numpy(np.float64)
if model.cluster_centers_ == None:
    model.cluster_centroids_ = cluster_centroids
    model.cluster_centers_ = cluster_centroids
else:
    pass

'--------------------------------------------------------'

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

#fpc visualization per num of clusters
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


'''VISUALIZATION IN 2D'''
# t-sne plotting for2D visualization (adjust t-SNE hyperparameters in the function def)
plot_tsne(X_train_new,clusters)
#k-medoids
plot_tsne(gd, clusters)
#fuzzy c-means
plot_tsne(X_train_reduc, clusters)

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

#fixing cluster memberships for fuzzy
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
X_test_n = X_test.drop(['age', 'sex','sub_labels'], axis=1)
num_list, cat_list = type_lists(X_test_n)


'----------------------------------------------------------------------------'

# Predicting the test sets based on strategy 1
X_test_pred1 = predict_whole_set(oversampled_clusters, X_test_n)

'----------------------------------------------'
# Predicting the test sets based on strategy 2
#K-PROTOTYPES:
costs = labels_cost(X_test_n, model.cluster_centroids_, euclidean_dissim,
                    matching_dissim, 0.5)
X_test_pred2 = predict_per_model(oversampled_clusters, model, X_test_n, costs,
                                 cat_list, unique_subl, test_sublabels)

#IF MODEL IS KMEDOIDS:
test_set = pd.concat([centroids_df, X_test_n], axis=0)
cats = [True if X_test_n[x].dtype == 'object' else False for x in X_test_n.columns]
gd_test = gower.gower_matrix(test_set, cat_features = cats)
pairwise_dist = gd_test[numc:, 0:numc]

X_test_pred2 = predict_w_clusters(oversampled_clusters, X_test_n, pairwise_dist,
                                  unique_subl, test_sublabels) 


#IF IT IS A FUZZY C-MEANS MODEL:
X_test_reduc = famd.transform(X_test_n)
#t_u is needed later for option 3
t_u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
    X_test_reduc.T, cntr, 2, error=0.005, maxiter=1000)
test_clusts = np.argmax(t_u, axis=0) 
X_test_pred2 = predict_w_fuzzy(oversampled_clusters, X_test_n, X_test_reduc, 
                                  cntr, unique_subl, test_sublabels) 

'----------------------------------------------'
# Predicting the test sets based on strategy 3
'''K-PROTOTYPES'''
X_test_pred3 = predict_w_weights(oversampled_clusters, costs, X_test_n,
                                 unique_subl, test_sublabels)

'''KMEDOIDS'''
X_test_pred3 = predict_w_weights_kmed(oversampled_clusters, pairwise_dist, 
                                      X_test_n, unique_subl, test_sublabels)
#NOTE: WEIGHTS GIVE ZERO DIVISION ERROR (due to some differences being 0)
co = 0
for item in range(len(pairwise_dist)):
    for i in range(numc):
        if pairwise_dist[item][i] == 0:
            co+=1
        else:
            pass

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