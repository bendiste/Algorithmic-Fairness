# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:27:27 2021

@author: Begum Hattatoglu
"""
import numpy as np
import pandas as pd

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import StructuredDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
    import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas

from imblearn.over_sampling import SMOTE, KMeansSMOTE, SMOTEN, SMOTENC

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


#Function to call the desired dataset with chosen features
def aif_data(dataset_used, protected_attribute_used, preprocessed):
    if dataset_used == "adult":
        if preprocessed == True:
            dataset_orig = load_preproc_data_adult()
            if protected_attribute_used == 1:
                privileged_groups = [{'sex': 1}]
                unprivileged_groups = [{'sex': 0}]
            else:
                privileged_groups = [{'race': 1}]
                unprivileged_groups = [{'race': 0}]
        else:
            dataset_orig = AdultDataset()
            if protected_attribute_used == 1:
                privileged_groups = [{'sex': 1}]
                unprivileged_groups = [{'sex': 0}]
            else:
                privileged_groups = [{'race': 1}]
                unprivileged_groups = [{'race': 0}]
        
    elif dataset_used == "german":
        if preprocessed == True:
            dataset_orig = load_preproc_data_german()
            if protected_attribute_used == 1:
                privileged_groups = [{'sex': 1}]
                unprivileged_groups = [{'sex': 0}]
            else:
                privileged_groups = [{'age': 1}]
                unprivileged_groups = [{'age': 0}]
            
            for i in range(1000):
                if (dataset_orig.labels[i] == 2.0):
                    dataset_orig.labels[i] = 0
                else:
                    dataset_orig.labels[i] = 1
            
            dataset_orig.favorable_label = 1
            dataset_orig.unfavorable_label = 0
            
        else:     
            dataset_orig = GermanDataset()
            if protected_attribute_used == 1:
                privileged_groups = [{'sex': 1}]
                unprivileged_groups = [{'sex': 0}]
            else:
                privileged_groups = [{'age': 1}]
                unprivileged_groups = [{'age': 0}]
                
            for i in range(1000):
                if (dataset_orig.labels[i] == 2.0):
                    dataset_orig.labels[i] = 0
                else:
                    dataset_orig.labels[i] = 1
                
            dataset_orig.favorable_label = 1
            dataset_orig.unfavorable_label = 0
          
    elif dataset_used == "compas":
        if preprocessed == True:
            dataset_orig = load_preproc_data_compas()
            if protected_attribute_used == 1:
                privileged_groups = [{'sex': 1}]
                unprivileged_groups = [{'sex': 0}]
            else:
                privileged_groups = [{'race': 1}]
                unprivileged_groups = [{'race': 0}]
        else:
            dataset_orig = CompasDataset()
            if protected_attribute_used == 1:
                privileged_groups = [{'sex': 1}]
                unprivileged_groups = [{'sex': 0}]
            else:
                privileged_groups = [{'race': 1}]
                unprivileged_groups = [{'race': 0}]
            
    return dataset_orig, privileged_groups, unprivileged_groups


'----------------------------------------------------------------------------'
#Function to create sublabel for the datasets
def sublabel(dataset_name, dataset_df):
    sub_labels = pd.Series()
    #Adult dataset sublabel construction
    if dataset_name == 'adult':
        for i in range(len(dataset_df)):
            #exp:black female
            if ((dataset_df['race'][i] == 0) & (dataset_df['sex'][i]==0) & (dataset_df['Income Binary'][i]==0)):
                sub_labels = sub_labels.append(pd.Series([0], index=[i]))
            elif ((dataset_df['race'][i] == 0) & (dataset_df['sex'][i]==0) & (dataset_df['Income Binary'][i]==1)):
                sub_labels = sub_labels.append(pd.Series([1], index=[i]))
            elif ((dataset_df['race'][i] == 1) & (dataset_df['sex'][i]==0) & (dataset_df['Income Binary'][i]==0)):
                sub_labels = sub_labels.append(pd.Series([2], index=[i]))
            elif ((dataset_df['race'][i] == 1) & (dataset_df['sex'][i]==0) & (dataset_df['Income Binary'][i]==1)):
                sub_labels = sub_labels.append(pd.Series([3], index=[i]))
            elif ((dataset_df['race'][i] == 0) & (dataset_df['sex'][i]==1) & (dataset_df['Income Binary'][i]==0)):
                sub_labels = sub_labels.append(pd.Series([4], index=[i]))
            elif ((dataset_df['race'][i] == 0) & (dataset_df['sex'][i]==1) & (dataset_df['Income Binary'][i]==1)):
                sub_labels = sub_labels.append(pd.Series([5], index=[i]))
            elif ((dataset_df['race'][i] == 1) & (dataset_df['sex'][i]==1) & (dataset_df['Income Binary'][i]==0)):
                sub_labels = sub_labels.append(pd.Series([6], index=[i]))
            #exp:white male
            elif ((dataset_df['race'][i] == 1) & (dataset_df['sex'][i]==1) & (dataset_df['Income Binary'][i]==1)):
                sub_labels = sub_labels.append(pd.Series([7], index=[i]))
            else:
                pass
    #German dataset sublabel construction
    elif dataset_name == 'german':
        for i in range(len(dataset_df)):
            if ((dataset_df['age'][i] == 0) & (dataset_df['sex'][i]==0) & (dataset_df['credit'][i]==0)):
                sub_labels = sub_labels.append(pd.Series([0], index=[i]))
            elif ((dataset_df['age'][i] == 0) & (dataset_df['sex'][i]==0) & (dataset_df['credit'][i]==1)):
                sub_labels = sub_labels.append(pd.Series([1], index=[i]))
            elif ((dataset_df['age'][i] == 1) & (dataset_df['sex'][i]==0) & (dataset_df['credit'][i]==0)):
                sub_labels = sub_labels.append(pd.Series([2], index=[i]))
            elif ((dataset_df['age'][i] == 1) & (dataset_df['sex'][i]==0) & (dataset_df['credit'][i]==1)):
                sub_labels = sub_labels.append(pd.Series([3], index=[i]))
            elif ((dataset_df['age'][i] == 0) & (dataset_df['sex'][i]==1) & (dataset_df['credit'][i]==0)):
                sub_labels = sub_labels.append(pd.Series([4], index=[i]))
            elif ((dataset_df['age'][i] == 0) & (dataset_df['sex'][i]==1) & (dataset_df['credit'][i]==1)):
                sub_labels = sub_labels.append(pd.Series([5], index=[i]))
            elif ((dataset_df['age'][i] == 1) & (dataset_df['sex'][i]==1) & (dataset_df['credit'][i]==0)):
                sub_labels = sub_labels.append(pd.Series([6], index=[i]))
            elif ((dataset_df['age'][i] == 1) & (dataset_df['sex'][i]==1) & (dataset_df['credit'][i]==1)):
                sub_labels = sub_labels.append(pd.Series([7], index=[i]))
            else:
                pass
    else:
        pass
        
    sub_labels = sub_labels.tolist()
    return sub_labels


'----------------------------------------------------------------------------'
#Function to output the baseline results of imbalanced datasets

def baseline_metrics(X_train, X_test, y_train, y_test):
    log_reg = LogisticRegression()
        
    X_train = X_train.loc[:, X_train.columns != 'sub_labels']
    X_test = X_test.loc[:, X_test.columns != 'sub_labels']  

    y_train = np.asarray(y_train).flatten()
    log_reg.fit(X_train, y_train)
    
    y_pred = log_reg.predict(X_test)
    X_test_res = pd.DataFrame(X_test.copy())
    X_test_res['y_pred'] = y_pred
    
    #later I an add an if check to adjust the groups based on the dataset given
    privileged_groups = [{'sex': 1, 'age': 1}, {'age':1}]
    unprivileged_groups = [{'sex': 0, 'age': 0}, {'age':0}]
    
    aif_binary_pred = BinaryLabelDataset(df=X_test_res, label_names=['y_pred'], 
                                            protected_attribute_names=['age', 'sex']) 
    aif_binary_orig = aif_binary_pred.copy()
    aif_binary_orig.labels = y_test
    
    ind = []
    results = {"AEO Difference": [], "Disparate Impact Ratio": [], "Dem Parity Difference": [],
           "Predictive Parity Difference": [], "Consistency": [],  "Accuracy": [], 
           "Balanced accuracy": [],  "F1-Score": [], "Precision (PPV)":[],
           "Recall (TPR)": [], "Specificity (TNR)":[]}
    
    #Construction 1
    #to construct this metric function, the predicted labels should be united with the test fetures to make a new datas
    metric_pred_test = BinaryLabelDatasetMetric(aif_binary_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    #Construction 2
    #both original test dataset with actual labels and the test dataset combined with predicted class labels need to be given to this function
    classified_metric = ClassificationMetric(aif_binary_orig, 
                                                     aif_binary_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
    
    #Checking Equalized Odds: average odds differecence, which is the avg. of differences in FPR&TPR for privileged and unprivileged groups.
    aeo = classified_metric.average_odds_difference()
    
    #Disparate Impact ratio between privileged and unprivileged groups.
    di = classified_metric.disparate_impact()
    
    #Demographic parity difference between privileged and unprivileged groups.
    spd = classified_metric.statistical_parity_difference()
    
    #Predictive parity difference: PPV difference between privileged and unprivileged groups.
    ppd = classified_metric.positive_predictive_value(privileged=False) - classified_metric.positive_predictive_value(privileged=True)
    
    #Individual Fairness: 1)Consistency, 2) Euclidean Distance between individuals.
    consistency = metric_pred_test.consistency()
    
    TPR = classified_metric.true_positive_rate() #recall
    TNR = classified_metric.true_negative_rate() #specificity
    PPV = classified_metric.positive_predictive_value() #precision
    bal_acc = (TPR+TNR)/2
    f1 = 2*((PPV*TPR)/(PPV+TPR))
    acc = classified_metric.accuracy()
    
    ind += ["Baseline model (LR)"]
    results["AEO Difference"].append(aeo)
    results["Disparate Impact Ratio"].append(di)
    results["Dem Parity Difference"].append(spd)
    results["Predictive Parity Difference"].append(ppd)
    results["Consistency"].append(consistency)
    results["Accuracy"].append(acc)
    results["Balanced accuracy"].append(bal_acc)
    results["F1-Score"].append(f1)
    results["Precision (PPV)"].append(PPV)
    results["Recall (TPR)"].append(TPR)
    results["Specificity (TNR)"].append(TNR)
              
    df_results = pd.DataFrame(results, index=ind)
    return df_results


'-----------------------------------------------------------------------------'
#Function that oversamples and classifies each cluster after applying the clustering algorithm
def oversample_german(X_train_new, X_test):
    cluster_dict = {}
    
    sm = SMOTEN(random_state=42, k_neighbors = 0)
    for i in range(len(X_train_new['cluster_labels'].unique())):
        cluster_dict[i] = X_train_new.loc[X_train_new['cluster_labels']==i]
        x_clust = cluster_dict[i]
        #clean cluster lanel since it is the same value for all instances
        x_clust = x_clust.drop(['cluster_labels','sub_labels'], axis=1)
        y_clust = cluster_dict[i]['sub_labels']
        y_clust=y_clust.astype('int')
            
        x_res, y_res = sm.fit_resample(x_clust, y_clust)
        y_label = x_res['class_labels']
        new_df = pd.concat([x_res, pd.DataFrame(y_res)], axis=1)
        cluster_dict[i] = new_df
        
        
    return cluster_dict


'-----------------------------------------------------------------------------'
#Option1 - using the whole oversampled training set

def predict_whole_set(cluster_dict, X_test):
    lr = LogisticRegression()
    df = pd.DataFrame()
    
    for i in range(len(cluster_dict)):
        data_part = cluster_dict[i]
        df = pd.concat([df,data_part], axis=0)
        
    df.reset_index(drop=True, inplace=True)   
    df = df.drop(['sub_labels'], axis=1)
    X_train = df.loc[:, df.columns != 'class_labels']
    y_train = df.loc[:, df.columns == 'class_labels']
    y_train = np.asarray(y_train).flatten()
    lr.fit(X_train, y_train)
    
    X_test = X_test.drop(['age', 'sex', 'sub_labels'], axis=1)
    y_pred = lr.predict(X_test)
    X_test_pred = pd.DataFrame(X_test.copy())
    X_test_pred['y_pred'] = y_pred
    
    return X_test_pred


'-----------------------------------------------------------------------------'
#Option2 - train model for each cluster, find each test sample's cluster to use correct model

def predict_per_model(cluster_dict, cluster_object, X_test, categorical_columns):
    lr = LogisticRegression()
    lr_models = {}

    for i in range(len(cluster_dict)):
        data_part = cluster_dict[i]
        data_part.reset_index(drop=True, inplace=True)   
        data_part = data_part.drop(['sub_labels'], axis=1)
        
        X_train = data_part.loc[:, data_part.columns != 'class_labels']
        y_train = data_part.loc[:, data_part.columns == 'class_labels']
        y_train = np.asarray(y_train).flatten()
        model = lr.fit(X_train, y_train) 
        lr_models[i] = model
        
    #find the right cluster the each sample belongs to
    X_test_n = X_test.drop(['age','sex','sub_labels'], axis=1)
    clusters = cluster_object.predict(X_test_n, categorical=categorical_columns)

    y_preds = []
    for i in range(len(X_test)):
        index = clusters[i]
        predicted=lr_models[index].predict(np.asarray(X_test_n.iloc[i,:]).reshape(1,-1))
        y_preds.append(predicted[0])
        
    X_test_n['y_pred'] = y_preds
    return X_test_n

'-----------------------------------------------------------------------------'
#Option3 - giving weights to each calculated model per cluster



'-----------------------------------------------------------------------------'
#Function to calculate the fairness metrics for given dataset(s)

def metrics_calculate(X_test, X_test_pred, y_test, prot_attr_names, privileged_groups, unprivileged_groups):
    ind = []
    results = {"AEO Difference": [], "Disparate Impact Ratio": [], "Dem Parity Difference": [],
           "Predictive Parity Difference": [], "Consistency": [],  "Accuracy": [], 
           "Balanced accuracy": [],  "F1-Score": [], "Precision (PPV)":[],
           "Recall (TPR)": [], "Specificity (TNR)":[]}

    #Transforming dataframes back to aif360 dataset (for german dataset)
    
    y = X_test_pred['y_pred']
    X_test = X_test.drop(['sub_labels'], axis=1)
    X_test['y_pred'] = y

    #transf. the dataframe into aif360 dataset type, then save in dyn name
    aif_binary = BinaryLabelDataset(df=X_test, label_names=['y_pred'], 
                                    protected_attribute_names=prot_attr_names) 
    
    #create original test set in aif360 format
    X_test_orig = aif_binary.copy()
    X_test_orig.labels = y_test
    
    #Construction 1
    #to construct this metric function, the predicted labels should be united with the test fetures to make a new datas
    metric_pred_test = BinaryLabelDatasetMetric(aif_binary, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    #Construction 2
    #both original test dataset with actual labels and the test dataset combined with predicted class labels need to be given to this function
    classified_metric = ClassificationMetric(X_test_orig, 
                                                     aif_binary,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)
    
    #Checking Equalized Odds: average odds differecence, which is the avg. of differences in FPR&TPR for privileged and unprivileged groups.
    aeo = classified_metric.average_odds_difference()
    
    #Disparate Impact ratio between privileged and unprivileged groups.
    di = classified_metric.disparate_impact()
    
    #Demographic parity difference between privileged and unprivileged groups.
    spd = classified_metric.statistical_parity_difference()
    
    #Predictive parity difference: PPV difference between privileged and unprivileged groups.
    ppd = classified_metric.positive_predictive_value(privileged=False) - classified_metric.positive_predictive_value(privileged=True)
    
    #Individual Fairness: 1)Consistency, 2) Euclidean Distance between individuals.
    consistency = metric_pred_test.consistency()
    
    TPR = classified_metric.true_positive_rate() #recall
    TNR = classified_metric.true_negative_rate() #specificity
    PPV = classified_metric.positive_predictive_value() #precision
    bal_acc = (TPR+TNR)/2
    f1 = 2*((PPV*TPR)/(PPV+TPR))
    acc = classified_metric.accuracy()
    
    ind += ["Implemented model"]
    results["AEO Difference"].append(aeo)
    results["Disparate Impact Ratio"].append(di)
    results["Dem Parity Difference"].append(spd)
    results["Predictive Parity Difference"].append(ppd)
    results["Consistency"].append(consistency)
    results["Accuracy"].append(acc)
    results["Balanced accuracy"].append(bal_acc)
    results["F1-Score"].append(f1)
    results["Precision (PPV)"].append(PPV)
    results["Recall (TPR)"].append(TPR)
    results["Specificity (TNR)"].append(TNR)
                  
    df_results = pd.DataFrame(results, index=ind)
    return df_results


'----------------------------------------------------------------------------'
