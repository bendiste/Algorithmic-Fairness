# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:12:06 2021

@author: hatta
"""
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes

def calc_euclian_dis(_s1, _s2):
    # s1 = np.array((3, 5))
    _eucl_dist = np.linalg.norm(_s2 - _s1)  # calculate Euclidean distance, accept input an array [2 6]
    return _eucl_dist
def calc_simpleMatching_dis(_s1, _s2):
    _cat_dist = 0
    if (_s1 != _s2):
        _cat_dist = 1
    return _cat_dist
k = 3
# calculate silhoutte for one cluster number
kproto = KPrototypes(n_clusters=k, init='Cao', verbose=2)
clusters_label = kproto.fit_predict(x_df_norm, categorical=[2])
_identical_cluster_labels = list(dict.fromkeys(clusters_label))
# Assign clusters lables to the Dataset
x_df_norm['Cluster_label'] = clusters_label
# ------------- calculate _silhouette_Index -------------
# 1. Calculate ai
_silhouette_Index_arr = []
for i in x_df_norm.itertuples():
    _ai_cluster_label = i[-1]
    # return samples of the same cluster
    _samples_cluster = x_df_norm[x_df_norm['Cluster_label'] == _ai_cluster_label]
    _dist_array_ai = []
    _s1_nume_att = np.array((i[1], i[2]))
    _s1_cat_att = i[3]
    for j in _samples_cluster.itertuples():
        _s2_nume_att = np.array((j[1], j[2]))
        _s2_cat_att = j[3]
        _euclian_dis = calc_euclian_dis(_s1_nume_att, _s2_nume_att)
        _cat_dis = calc_simpleMatching_dis(_s1_cat_att, _s2_cat_att)
        _dist_array_ai.append(_euclian_dis + (kproto.gamma * _cat_dis))
    ai = np.average(_dist_array_ai)
    # 2. Calculate bi
    # 2.1. determine the samples of other clusters
    _identical_cluster_labels.remove(_ai_cluster_label)
    _dic_cluseter = {}
    _bi_arr = []
    for ii in _identical_cluster_labels:
        _samples = x_df_norm[x_df_norm['Cluster_label'] == ii]
        # 2.2. calculate bi
        _dist_array_bi = []
        for j in _samples.itertuples():
            _s2_nume_att = np.array((j[1], j[2]))
            _s2_cat_att = j[3]
            _euclian_dis = calc_euclian_dis(_s1_nume_att, _s2_nume_att)
            _cat_dis = calc_simpleMatching_dis(_s1_cat_att, _s2_cat_att)
            _dist_array_bi.append(_euclian_dis + (kproto.gamma * _cat_dis))
        _bi_arr.append(np.average(_dist_array_bi))
    _identical_cluster_labels.append(_ai_cluster_label)
    # min bi is determined as final bi variable
    bi = min(_bi_arr)
    # 3. calculate silhouette Index
    if ai == bi:
        _silhouette_i = 0
    elif ai < bi:
        _silhouette_i = 1 - (ai / bi)
    elif ai > bi:
        _silhouette_i = 1 - (bi / ai)
    _silhouette_Index_arr.append(_silhouette_i)
silhouette_score = np.average(_silhouette_Index_arr)
print('_silhouette_Index = ' + str(silhouette_score))





'---------------------------------------------------------------------------'

#MY SILHOUETTE SCORE

#preparing the distance matrix
from sklearn.feature_extraction import DictVectorizer
dist_dict = costs = labels_cost(X_train_new, kprot.cluster_centroids_, 
                             euclidean_dissim, matching_dissim, 0.5)
data = dist_dict.items()
matr = list(data)

vertical_array = np.empty((0, 9), int)
for i in matr:
    vertical_array = np.append(vertical_array, np.array([i[1]]), axis=0)
dist_m = np.asmatrix(vertical_array)


silhouette_avg = silhouette_score(dist_m, clusters)
print("For n_clusters =", nc,"The average silhouette_score is :", 
      silhouette_avg)
# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(dist_m, clusters)
y_lower = 10
for i in range(nc):
    # Aggregate the silhouette scores for samples belonging to
    # cluster i, and sort them
    ith_cluster_silhouette_values = \
        sample_silhouette_values[kprot == i]

    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / nc)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    # Label the silhouette plots with their cluster numbers at the middle
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    # Compute the new y_lower for next plot
    y_lower = y_upper + 10  # 10 for the 0 samples

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

# 2nd Plot showing the actual clusters formed
colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
            c=colors, edgecolor='k')

# Labeling the clusters
centers = clusterer.cluster_centers_
# Draw white circles at cluster centers
ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
            c="white", alpha=1, s=200, edgecolor='k')

for i, c in enumerate(centers):
    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                s=50, edgecolor='k')

ax2.set_title("The visualization of the clustered data.")
ax2.set_xlabel("Feature space for the 1st feature")
ax2.set_ylabel("Feature space for the 2nd feature")

plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
              "with n_clusters = %d" % n_clusters),
             fontsize=14, fontweight='bold')

