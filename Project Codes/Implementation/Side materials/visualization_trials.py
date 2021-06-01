# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 14:49:39 2021

@author: hatta
"""
import pandas as pd
import numpy as np
import gower

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import KernelPCA

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist,squareform

import matplotlib.pyplot as plt
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D

'------------------------------------------------------------------------'

#get categorical columns
cat = [True if X_train_new[x].dtype == 'object' else False for x in X_train_new.columns]

gd = gower.gower_matrix(X_train_new, cat_features = cat)
#transform the distance matrix to 1D in order to use scipy hc function: linkage
gdv = squareform(gd,force='tovector')
#HAC custering
Z = linkage(gdv,method='ward')
Z_df = pd.DataFrame(Z,columns=['id1','id2','dist','n'])

# GETTING THE CLUSTER LABEL PER SAMPLE
# Form flat clusters from the hierarchical clustering defined by the given linkage matrix.
cld = fcluster(Z, 4, criterion='maxclust')

# Dendrogram visualization
fig,axs = plt.subplots(1,1,figsize=(30,6))
dn = dendrogram(Z, truncate_mode='level',p=6,show_leaf_counts=True,ax=axs);
print(f"Leaves = {len(dn['leaves'])}")


#PLOTTING
''' Silhouette scores plot per k clusters'''
# find k clusters
results = dict()
k_cand = list(np.arange(5,55,5))
k_cand.extend(list(np.arange(50,500,50)))

for k in k_cand:
    cluster_array = fcluster(Z, k, criterion='maxclust')
    score0 = silhouette_score(gd, cluster_array, metric='precomputed')
    score1 = silhouette_score(X_train_new, cluster_array,metric='cityblock')
    results[k] = {'k':cluster_array,'s0':score0,'s1':score1}
    
fig,axs = plt.subplots(1,1,figsize=(16,5))
axs.plot([i for i in results.keys()],[i['s0'] for i in results.values()],'o-',label='Gower')
axs.plot([i for i in results.keys()],[i['s1'] for i in results.values()],'o-',label='Cityblock')
axs.set_xlim(1,451)
axs.set_xticks(k_cand)
axs.set_xlabel('K')
axs.legend();

'-----------------------------------------------------------------------------'

# Applying Kernel PCA 
kpca = KernelPCA(n_components = 2, kernel = 'cosine', fit_inverse_transform = False)
X = kpca.fit_transform(X_train_new)

# Plot first 2 principal components 2D (n_components = 2)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
xs = X[:,0]
ys = X[:,1]
ax.scatter(xs,ys)
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_title('The first 2 principal component of KPCA, kernel = cosine', fontsize = 15)
ax.grid()


# Plot first three principal components in 3D (n_components = 3)
fig = plt.figure()
ax = Axes3D(fig)
#ax = fig.add_subplot(111,projection='3d')
xs = X[:,0]
ys = X[:,1]
zs = X[:,2]
ax.scatter(xs,ys,zs, alpha=0.5, cmap='spring')
ax.set_axis_bgcolor("lightgrey")
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.set_zlabel('Third Principal Component')
plt.show()

'----------------------------------------------------------------------------'
# Silhouette score and plots for clusters of KMeans
k_cand = [4, 5, 6, 7, 8]

fig,axs = plt.subplots(len(k_cand),2,figsize=(12,12))

for e,k in enumerate(k_cand):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
  
    cdict = {i:cm.Set1(i) for i in np.unique(kmeans.labels_)}
    
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    print("For n_clusters =", k,
          "The average silhouette_score is :", silhouette_avg)
    
    ## silhouette samples
    silhouette_vals = silhouette_samples(X,kmeans.labels_)
    y_lower = 0 
    y_upper = 0
    for i,cluster in enumerate(np.unique(kmeans.labels_)):
        cluster_silhouette_vals = silhouette_vals[kmeans.labels_==cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        axs[e,0].barh(range(y_lower,y_upper),
                      cluster_silhouette_vals,
                      height=1,
                      color=cdict[cluster])
        axs[e,0].text(-0.03,(y_lower+y_upper)/2,str(i))
        
        y_lower += len(cluster_silhouette_vals) # add for next iteration 
        avg_score = np.mean(silhouette_vals)
        axs[e,0].axvline(avg_score,linestyle ='--',color = 'black')
        
        axs[e,0].set_yticks([])
        axs[e,0].set_xlim([-0.1, 1])
        axs[e,0].set_xlabel('Silhouette coefficient values')
        axs[e,0].set_ylabel('Cluster labels')
        axs[e,0].set_title('Silhouette plot for the various clusters')
        
    ## plot data and cluster centroids
    results = pd.DataFrame(X)
    results['k'] = kmeans.labels_
    for cluster in np.unique(kmeans.labels_): # plot data by cluster
        axs[e,1].scatter(x=results.where(results['k']==cluster)[0],
                          y=results.where(results['k']==cluster)[1],
                          color=cdict[cluster],
                          label=cluster)

    # plot centroids
    axs[e,1].scatter(x=kmeans.cluster_centers_[:,0],
                      y=kmeans.cluster_centers_[:,1],
                      marker='x',color='black',s=180)
    axs[e,1].legend(bbox_to_anchor=(1,1))
    axs[e,1].set_title(f"kmeans\n$k$ = {k}")
    plt.tight_layout()

'---------------------------------------------------------------------------'
#Silhouette visualization for HAC
'''' NOTE: KPCA version is given as input, for original, give distance matrix'''
k_cand = [4, 5, 6, 7, 8]

fig,axs = plt.subplots(len(k_cand),2,figsize=(12,12))

for e,k in enumerate(k_cand):
    clusterer = AgglomerativeClustering(n_clusters=k, affinity='manhattan', 
                                        linkage='average')
    cluster_labels = clusterer.fit_predict(X)
  
    cdict = {i:cm.Set1(i) for i in np.unique(cluster_labels)}
    
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", k,
          "The average silhouette_score is :", silhouette_avg)
    
    ## silhouette samples
    silhouette_vals = silhouette_samples(X,cluster_labels)
    y_lower = 0 
    y_upper = 0
    for i,cluster in enumerate(np.unique(cluster_labels)):
        cluster_silhouette_vals = silhouette_vals[cluster_labels==cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        axs[e,0].barh(range(y_lower,y_upper),
                      cluster_silhouette_vals,
                      height=1,
                      color=cdict[cluster])
        axs[e,0].text(-0.03,(y_lower+y_upper)/2,str(i))
        
        y_lower += len(cluster_silhouette_vals) # add for next iteration 
        avg_score = np.mean(silhouette_vals)
        axs[e,0].axvline(avg_score,linestyle ='--',color = 'black')
        
        axs[e,0].set_yticks([])
        axs[e,0].set_xlim([-0.1, 1])
        axs[e,0].set_xlabel('Silhouette coefficient values')
        axs[e,0].set_ylabel('Cluster labels')
        axs[e,0].set_title('Silhouette plot for the various clusters')
        
    ## plot data and cluster centroids
    results = pd.DataFrame(X)
    results['k'] = cluster_labels
    for cluster in np.unique(cluster_labels): # plot data by cluster
        axs[e,1].scatter(x=results.where(results['k']==cluster)[0],
                         y=results.where(results['k']==cluster)[1],
                         color=cdict[cluster],
                         label=cluster)

    axs[e,1].legend(bbox_to_anchor=(1,1))
    axs[e,1].set_title(f"HAC\n$k$ = {k}")
    plt.tight_layout()



'---------------------------------------------------------------------------'

#Finally, get the cluster labels of each sample with sklearn version of agglomerative hc
nc = 6
model = AgglomerativeClustering(n_clusters = nc, affinity= 'manhattan', 
                             linkage= 'average')
clustering = model.fit(X)
clusters = clustering.labels_
#or, to obtain the labels:
results = model.fit_predict(gd)
dendrogram(clustering)


'--------------------------------------------------------------------------'

