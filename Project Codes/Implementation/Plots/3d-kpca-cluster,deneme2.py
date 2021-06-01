# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:44:15 2018

@author: begum.hattatoglu
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset with pandas
#Inputta ürün kategorisi tercihi var!
dataset = pd.read_excel('INPUT.xlsx')

#Applying feature scaling on the numeric feature
from sklearn.preprocessing import StandardScaler
scaled_feature = dataset.copy()
col_name = ['Average_Order_Fee']
feature = scaled_feature[col_name]
scaler = StandardScaler().fit(feature.values)
feature = scaler.transform(feature.values)
scaled_feature[col_name] = feature

X = scaled_feature.iloc[:, 1:24].values
df = pd.DataFrame(X)

#obtaining gower distances of instances
import gower_functionv6 as gf
Gower = gf.gower_distances(X)

# Applying Kernel PCA 
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'sigmoid', fit_inverse_transform = True)
X = kpca.fit_transform(Gower)

#Kernel values for detailed explanation of the results
ALPHAS = kpca.alphas_
LAMBDAS = kpca.lambdas_
DUALCOEF = kpca.dual_coef_
kpca.X_fit_
Projection = kpca.X_transformed_fit_


# Plot first 2 principal components 2D (n_components = 2)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
xs = X[:,0]
ys = X[:,1]
ax.scatter(xs,ys)
ax.set_xlabel('Principal Component 1', fontsize = 10)
ax.set_ylabel('Principal Component 2', fontsize = 10)
ax.set_title('The first 2 principal component of KPCA, kernel = sigmoid', fontsize = 15)
ax.grid()


# Plot first three principal components in 3D (n_components = 3)
from mpl_toolkits.mplot3d import Axes3D
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


#using dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'average'))
# ward method is the method that tries to minimize the variance within each cluster
plt.title('dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


#fitting hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
# Vector of clusters : hc, y_hc
#n_clusters is changed to the optimal num of cluster after the silhouette score result
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'complete', linkage = 'average')
y_hc = hc.fit_predict(X)
dataf = pd.DataFrame(X)

#writing the cluster labels into an excel file
clusters = pd.DataFrame(y_hc)
clusters.to_excel('clusters.xlsx', sheet_name='sheet1', index=False)

#Plotting the clusters derşved from the first two principal components in 2D
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c= 'red', label = 'Cluester1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c= 'blue', label = 'Cluester2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c= 'green', label = 'Cluester3')
#>> if needed
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c= 'cyan', label = 'Cluester4') 
#>> if needed
#plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c= 'magenta', label = 'Cluester5') 
plt.title('Clusters of Clients with KPCA preprocessing with HAC, using Gower Distance Metric')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

#concatenation of arrays on a dataframe and then converting back to a numpy array (to prepare 3D)
df2 = pd.DataFrame(X)
df3 = pd.DataFrame(y_hc)
df4 = pd.concat([df2, df3], axis=1)
numpy_matrix = df4.as_matrix()

# Plot the clusters derived from the first three principal components in 3D
from mpl_toolkits.mplot3d import Axes3D

df = pd.DataFrame(numpy_matrix, columns=['0', '1','2', '3'])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(df['0'])
y = np.array(df['1'])
z = np.array(df['2'])

ax.set_axis_bgcolor("lightgrey")
ax.set_xlabel('First Principal Component')
ax.set_ylabel('Second Principal Component')
ax.set_zlabel('Third Principal Component')
plt.title('Clusters of Clients (Hierarchical Agglomerative Clustering with KPCA)')
ax.scatter(x,y,z, marker="o", c=df['3'], s=100, edgecolor = 'k')

plt.show()


#Silhouette Score calculation & visualization
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score

range_n_clusters = [3, 4, 5, 6, 7, 8]
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    from sklearn.cluster import AgglomerativeClustering
    clusterer = AgglomerativeClustering(n_clusters = n_clusters, affinity = 'cosine',
                                        linkage = 'average')
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        cmap = cm.get_cmap("Spectral")
        color = cmap(float(i) / n_clusters)
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
    cmap = cm.get_cmap("Spectral")
    colors = cmap(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for HAC clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.show()