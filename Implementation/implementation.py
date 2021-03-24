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

import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd

import umap #special dimensionality reduction
import umap.plot #in order to plot k-prototypes

from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes

from prince import FAMD #Factor analysis of mixed data

from imblearn.over_sampling import SMOTE, KMeansSMOTE, SMOTEN, SMOTENC
from imblearn.pipeline import make_pipeline
from imblearn import FunctionSampler  # to use a idendity sampler

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StructuredDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns


## import dataset
'''dataset_used = "adult", "german", "compas"
   protected_attribute_used = 1,2 (check the function for the correct sens. attr.)
   preprocessed_dataset = True, original_dataset = False'''
        
dataset_prep, privileged_groups, unprivileged_groups = aif_data("german", 2, True)
dataset_orig, privileged_groups, unprivileged_groups = aif_data("german", 2, False)
dataset_used = "german" #assign the dataset name to an obj to use as var. later

# Initial disparities in the original dataset based on fairness metrics
metric_orig = BinaryLabelDatasetMetric(dataset_orig, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

print("Disparate impact (of original labels) between unprivileged and privileged groups = %f" % metric_orig.disparate_impact())
print("Difference in statistical parity (of original labels) between unprivileged and privileged groups = %f" % metric_orig.statistical_parity_difference())
print("Individual fairness metric from Zemel et.al. that measures how similar the labels are for similar instances = %f" % metric_orig.consistency())

'----------------------------------------------------------------------------'
#creating the snythetic sub-class label column on dataframe

#pre-processed
prep = dataset_prep.convert_to_dataframe()
prep_df = prep[0]

#original dataset
orig = dataset_orig.convert_to_dataframe()
orig_df = orig[0]

'----------------------------------------------------------------------------'
#if the imported dataset is Adult!
orig_df = orig_df.loc[:,['capital-gain','capital-loss','hours-per-week']]
dataset_df = pd.concat([prep_df, orig_df], axis=1, join = 'inner')

#if you also use the original adult data only: to get some of the num. features
orig_df1 = orig_df.iloc[:,0:57]
label = orig_df.iloc[:,98].copy()
orig_df1['Income Binary'] = label
orig_df = orig_df1.drop(['education-num'], axis=1)

'----------------------------------------------------------------------------'
#Note: pre-processed data, sublabel func doesn't have compas dataset implemented yet.
sub_labels = sublabel(dataset_used, dataset_df)
prep_df['sub_labels'] = sub_labels
subgroup_sizes = prep_df['sub_labels'].value_counts()

#or if the dataset is NOT the pre-processed version:
sub_labels = sublabel(dataset_used, orig_df)
orig_df['sub_labels'] = sub_labels
subgroup_sizes = orig_df['sub_labels'].value_counts()

'----------------------------------------------------------------------------'
#stratify=sub_labels or y?
#Train-test split
X = prep_df.loc[:, prep_df.columns != 'credit']
y = prep_df.loc[:, prep_df.columns == 'credit'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
        
                                                    random_state=42, stratify=sub_labels)
#for NOT pre-processed dataset
X = orig_df.loc[:, orig_df.columns != 'credit']
y = orig_df.loc[:, orig_df.columns == 'credit'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42, stratify=sub_labels)

#check class imbalance in the splitted training set
X_train['sub_labels'].value_counts()
X_test['sub_labels'].value_counts()

#FEATURE SCALING
#scale the numerical columns of the X_train set 
scale = StandardScaler()
#assign data type as object to the categorical variables 
X_train[X_train.columns[7:59]] = X_train[X_train.columns[7:59]].astype('object')
X_train[X_train.columns[4]] = X_train[X_train.columns[4]].astype('object')

x_cat = X_train.drop(['month','credit_amount','investment_as_income_percentage', 
                      'number_of_credits', 'residence_since', 'people_liable_for'],
                       axis = 1)
x_cat.reset_index(drop=True, inplace=True)
x_num = X_train.select_dtypes(exclude='object')
x_num = pd.DataFrame(scale.fit_transform(x_num))
#concat back the numerical and categorial columns as training set
X_train = pd.concat([x_num,x_cat], axis=1)

'----------------------------------------------------------------------------'

#Calculate the base metrics from the imbalanced dataset

#Note: the function is created based on the assump. that the X's have sub_labels
baseline_stats = baseline_metrics(X_train, X_test, y_train, y_test)

#if I want to see results as one column
baseline_stats = baseline_stats.T

'-----------------------------------------------------------------------------'

#clustering implementation on the dataset
keep_sub_l = X_train['sub_labels']
#Required drops for german dataset
X_train_new = X_train.drop(['age', 'sex', 'sub_labels'], axis=1)

#for k-prototypes
categorical_columns = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
                       25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,
                       43,44,45,46,47,48,49,50,51]
'''[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
                       24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,
                       43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59]'''

'-----------------------------------------------------------------------------'
    
#Dimensionality reduction for big datasets (adult)

X_train_new['sub_labels'] = keep_sub_l
cat_cols = [col for col in X_train_new.columns if col not in ['capital-gain',
                                                              'capital-loss', 
                                                              'hour-per-week']]
#for German
cat_cols = [col for col in X_train_new.columns if col not in ['month','credit_amount','investment_as_income_percentage', 
                                                         'number_of_credits', 'people_liable_for']]

for col in cat_cols:
    X_train_new[col] = X_train_new[col].astype(str)


famd = FAMD(n_components =2, n_iter = 3, random_state = 42)
famd.fit(X_train_new.drop('sub_labels', axis=1))
X_train_reduc = famd.transform(X_train_new)
ax = famd.plot_row_coordinates(X_train_new, 
                               color_labels=['sub-labels {}'.format(t) for t in X_train_new['sub_labels']] )
X_train_red = famd.partial_row_coordinates(X_train_new)
famd.explained_inertia_
ax = famd.plot_partial_row_coordinates(X_train_new, 
                                       color_labels=['sub-labels {}'.format(t) for t in X_train_new['sub_labels']])
X_train_new = X_train.drop(['sub_labels'], axis=1)

'----------------------------------------------------------------------------'

#X_train_new = X_train_new.drop(['sub_labels'], axis=1)

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

#putting the required label info back to the dataframe
X_train_new['cluster_labels'] = cluster_labels
X_train_new['sub_labels'] = keep_sub_l
X_train_new['class_labels'] = y_train

'----------------------------------------------------------------------------'

#Elbow method for kprototypes
costs = []
n_clusters = []
clusters_assigned = []
from tqdm import tqdm
for i in tqdm(range(5, 15)):
    try:
        cluster = KPrototypes(n_clusters=i, init='Cao')
        clusters = cluster.fit_predict(X_train_new, categorical=categorical_columns)
        costs.append(cluster.cost_)
        n_clusters.append(i)
        clusters_assigned.append(clusters)
    except:
        print(f"Can't cluster with {i} clusters")
       
plt.scatter(x=n_clusters, y=costs)
plt.plot(n_clusters, costs)
plt.show()

#actual clustering with k-prototypes
kprot = KPrototypes(n_clusters=10, init='Cao')
clusters = kprot.fit_predict(X_train_new, categorical=categorical_columns)


#VISUALIZATION

#change the data types of categorical variables
X_train_new[X_train_new.columns[6:51]] = X_train_new[X_train_new.columns[6:51]].astype('object')

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

#putting the required label info back to the dataframe
X_train_new['cluster_labels'] = clusters
X_train_new['sub_labels'] = keep_sub_l
X_train_new['class_labels'] = y_train

'----------------------------------------------------------------------------'

#over-sampling of each cluster

cluster_datasets, X_test_preds, y_preds = oversample_german(X_train_new, X_test)

'----------------------------------------------------------------------------'
#predicting the training based on their cluster membership(?)


'Question regarding classification is in the function code.'

'----------------------------------------------------------------------------'

#german dataset metrics table creation
prot_attrs = ['age', 'sex']
privileged_groups = [{'sex': 1, 'age': 1}, {'age':1}]
unprivileged_groups = [{'sex': 0, 'age': 0}, {'age': 0}]
metrics_table = metrics_calculate(X_test_preds, y_test, prot_attrs, 
                                  privileged_groups, unprivileged_groups)

