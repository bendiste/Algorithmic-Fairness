# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:40:10 2021

@author: hatta
"""

from kmodes.kprototypes import *
from kmodes.util.dissim import *
from kmodes.util import *

#Elbow method for kmodes
costs = []
n_clusters = []
clusters_assigned = []
from tqdm import tqdm
for i in tqdm(range(2, 20)):
    try:
        cluster = KModes(n_clusters= i, init='Huang', verbose=1)
        clusters = cluster.fit_predict(X_train_new)
        costs.append(cluster.cost_)
        n_clusters.append(i)
        clusters_assigned.append(clusters)
    except:
        print(f"Can't cluster with {i} clusters")
       
plt.scatter(x=n_clusters, y=costs)
plt.plot(n_clusters, costs)
plt.show()

'QUESTION: Should I pick the top n clusters with best silhouette scores?'

#actual clustering with k-modes
kmodes = KModes(n_clusters=6, init='Cao', verbose=1)
kmodes.fit_predict(X_train_new)
print(kmodes.cluster_centroids_)
cluster_labels = kmodes.labels_