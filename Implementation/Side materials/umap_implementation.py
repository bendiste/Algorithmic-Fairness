# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 12:31:34 2021

@author: hatta
"""

#VISUALIZATION with umap
#Preprocessing numerical variables
numerical = X_train_new.select_dtypes(exclude='object')
#Percentage of columns which are categorical is used as weight parameter in embeddings later
categorical_weight = len(X_train_new.select_dtypes(include='object').columns) / X_train_new.shape[1]

for c in numerical.columns:
    pt = PowerTransformer()
    numerical.loc[:, c] = pt.fit_transform(np.array(numerical[c]).reshape(-1, 1))
##preprocessing categorical variables
categorical = X_train_new.select_dtypes(include='object')
categorical = pd.get_dummies(categorical)
#visualizing the clusters
fit1 = umap.UMAP(metric='l2').fit(numerical)
fit2 = umap.UMAP(metric='dice').fit(categorical)

#plot only numerical variables
umap.plot.points(fit1)
#plot only categorical variables
umap.plot.points(fit2)

#plot the combination of categ and numeric variables
intersection = umap.umap_.general_simplicial_set_intersection(fit1.graph_, fit2.graph_, weight=categorical_weight)
intersection = umap.umap_.reset_local_connectivity(intersection)
embedding = umap.umap_.simplicial_set_embedding(fit1._raw_data, intersection, fit1.n_components, 
                                                fit1._initial_alpha, fit1._a, fit1._b, 
                                                fit1.repulsion_strength, fit1.negative_sample_rate, 
                                                200, 'random', np.random, fit1.metric, 
                                                fit1._metric_kwds, densmap = False, densmap_kwds={},
                                                output_dens=False)

fig, ax = plt.subplots()
fig.set_size_inches((20, 10))
scatter = ax.scatter(embedding[0][:, 0], embedding[0][:, 1], s=2, c=clusters, cmap='tab20b', alpha=1.0)

# produce a legend with the unique colors from the scatter
legend1 = ax.legend(*scatter.legend_elements(num=7),
                    loc="lower left", title="Clusters")
ax.add_artist(legend1)