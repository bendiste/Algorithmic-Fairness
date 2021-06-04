# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 11:47:54 2021

@author: hatta
"""
#the clustering algs that didn't work

'''DBSCAN'''
from sklearn.cluster import DBSCAN
#find the optimal epsilon value with nearest neighbors plot
from sklearn.neighbors import NearestNeighbors
#k(nearest neighbors) = minPts-1
neighbors = NearestNeighbors(n_neighbors=17)
neighbors_fit = neighbors.fit(X_train_reduc)
distances, indices = neighbors_fit.kneighbors(X_train_reduc)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)
epsilon = 0.78
# minimum points is the number of features in the dataset +1 or 2*dim (18 dims)
minpts = 19
model = DBSCAN(eps=epsilon, min_samples=minpts)
clusters = model.fit_predict(X_train_reduc)
num_clusters = len(set(clusters))
#-1 means noise
print(num_clusters)

'''HDBSCAN'''
import hdbscan
model = hdbscan.HDBSCAN(min_cluster_size=12, prediction_data=True).fit(X_train_red)
clusters = model.labels_
# If we wish to know which branches were selected by the HDBSCAN* algorithm we can pass select_clusters=True.
model.condensed_tree_.plot(select_clusters=True,
                               selection_palette=sns.color_palette('deep', 8))

# after fixing cluster memberships
'''if it is dbscan model'''
dict1 = {0: fixed_clusters[0]} #then use dict 1 instead of fixed_clusters to oversample



'''PREDICTING WITH THESE MODELS'''
#DBSCAN MODEL
import scipy as sp
def dbscan_predict(dbscan_model, X_new):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int)*-1 

    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_): 
            if abs(x_new - x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break

    return y_new

X_test_clusts = dbscan_predict(model, gd_tes2)

#HDBSCAN MODEL
#to predict new cluster labels from HDBSCAN for a new dataset
test_labels, strengths = hdbscan.approximate_predict(model, test_points)
