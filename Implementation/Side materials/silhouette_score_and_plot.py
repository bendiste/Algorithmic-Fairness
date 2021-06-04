# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 10:12:06 2021

@author: hatta
"""
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes

'''SILHOUETTE SCORE FOR K-PROTOTYPES'''

def calc_euclian_dis(_s1, _s2):
    # s1 = np.array((3, 5))
    _eucl_dist = np.linalg.norm(_s2 - _s1)  # calculate Euclidean distance, accept input an array [2 6]
    return _eucl_dist
def calc_simpleMatching_dis(_s1, _s2):
    _cat_dist = 0
    if (_s1 != _s2):
        _cat_dist = 1
    return _cat_dist
k = 6
# calculate silhoutte for one cluster number
kproto = KPrototypes(n_clusters=k, init='Cao', verbose=2)
clusters_label = kproto.fit_predict(X_train_new, categorical=cat_list)
_identical_cluster_labels = list(dict.fromkeys(clusters_label))
# Assign clusters lables to the Dataset
X_train_new['Cluster_label'] = clusters_label

# ------------- calculate _silhouette_Index -------------
# 1. Calculate ai
_silhouette_Index_arr = []
for i in X_train_new.itertuples():
    _ai_cluster_label = i[-1]
    # return samples of the same cluster
    _samples_cluster = X_train_new[X_train_new['Cluster_label'] == _ai_cluster_label]
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
        _samples = X_train_new[X_train_new['Cluster_label'] == ii]
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
