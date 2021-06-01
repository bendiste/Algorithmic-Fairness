# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:27:27 2021

@author: Begum Hattatoglu
"""
import numpy as np
import pandas as pd

from warnings import simplefilter
from collections import Counter

from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
    import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas

from kmodes.kprototypes import *
from kmodes.kmodes import *
from kmodes.util.dissim import matching_dissim, ng_dissim, euclidean_dissim

from deepcopy import *

from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN 

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt

#Function to call the desired dataset with chosen features
'''dataset_used = "adult", "german", "compas"
   protected_attribute_used = 1,2 (check the aif360 docs for the correct sens. attr.)
   preprocessed_dataset = True, original_dataset = False'''
def aif_data(dataset_used, preprocessed):
    if dataset_used == "adult":
        if preprocessed == True:
            dataset_orig = load_preproc_data_adult()
            privileged_groups = [{'sex': 1,'race': 1}]
            unprivileged_groups = [{'sex': 0,'race': 0}]

        else:
            dataset_orig = AdultDataset()
            privileged_groups = [{'sex': 1,'race': 1}]
            unprivileged_groups = [{'sex': 0,'race': 0}]

        
    elif dataset_used == "german":
        if preprocessed == True:
            dataset_orig = load_preproc_data_german()
            privileged_groups = [{'sex': 1,'age': 1}]
            unprivileged_groups = [{'sex': 0,'age': 0}]
            
            for i in range(1000):
                if (dataset_orig.labels[i] == 2.0):
                    dataset_orig.labels[i] = 0
                else:
                    dataset_orig.labels[i] = 1
            
            dataset_orig.favorable_label = 1
            dataset_orig.unfavorable_label = 0
            
        else:     
            dataset_orig = GermanDataset()
            privileged_groups = [{'sex': 1,'age': 1}]
            unprivileged_groups = [{'sex': 0,'age': 0}]
                
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
            privileged_groups = [{'sex': 1,'race': 1}]
            unprivileged_groups = [{'sex': 0,'race': 0}]

        else:
            dataset_orig = CompasDataset()
            privileged_groups = [{'sex': 1,'race': 1}]
            unprivileged_groups = [{'sex': 0,'race': 0}]
            
    return dataset_orig, privileged_groups, unprivileged_groups


'----------------------------------------------------------------------------'
#Function to create sublabel for the datasets
def sublabel(dataset_df, sens_attr, decision_label):
    sub_labels = pd.Series(dtype='object')

    #sensitive attribute(s) must be given as a list
    if len(sens_attr) == 2:
        for i in range(len(dataset_df)):
            if ((dataset_df[sens_attr[0]].iloc[i] == 0) & (dataset_df[sens_attr[1]].iloc[i]==0) & (dataset_df[decision_label].iloc[i]==0)):
                sub_labels = sub_labels.append(pd.Series([0], index=[i]))
            elif ((dataset_df[sens_attr[0]].iloc[i] == 0) & (dataset_df[sens_attr[1]].iloc[i]==0) & (dataset_df[decision_label].iloc[i]==1)):
                sub_labels = sub_labels.append(pd.Series([1], index=[i]))
            elif ((dataset_df[sens_attr[0]].iloc[i] == 1) & (dataset_df[sens_attr[1]].iloc[i]==0) & (dataset_df[decision_label].iloc[i]==0)):
                sub_labels = sub_labels.append(pd.Series([2], index=[i]))
            elif ((dataset_df[sens_attr[0]].iloc[i] == 1) & (dataset_df[sens_attr[1]].iloc[i]==0) & (dataset_df[decision_label].iloc[i]==1)):
                sub_labels = sub_labels.append(pd.Series([3], index=[i]))
            elif ((dataset_df[sens_attr[0]].iloc[i] == 0) & (dataset_df[sens_attr[1]].iloc[i]==1) & (dataset_df[decision_label].iloc[i]==0)):
                sub_labels = sub_labels.append(pd.Series([4], index=[i]))
            elif ((dataset_df[sens_attr[0]].iloc[i] == 0) & (dataset_df[sens_attr[1]].iloc[i]==1) & (dataset_df[decision_label].iloc[i]==1)):
                sub_labels = sub_labels.append(pd.Series([5], index=[i]))
            elif ((dataset_df[sens_attr[0]].iloc[i] == 1) & (dataset_df[sens_attr[1]].iloc[i]==1) & (dataset_df[decision_label].iloc[i]==0)):
                sub_labels = sub_labels.append(pd.Series([6], index=[i]))
            elif ((dataset_df[sens_attr[0]].iloc[i] == 1) & (dataset_df[sens_attr[1]].iloc[i]==1) & (dataset_df[decision_label].iloc[i]==1)):
                    sub_labels = sub_labels.append(pd.Series([7], index=[i])) 
    #if there is a single binary attribute
    elif len(sens_attr) == 1:
        for i in range(len(dataset_df)):
            if ((dataset_df[sens_attr[0]].iloc[i] == 0) & (dataset_df[decision_label].iloc[i]==0)):
                sub_labels = sub_labels.append(pd.Series([0], index=[i]))
            elif ((dataset_df[sens_attr[0]].iloc[i] == 0) & (dataset_df[decision_label].iloc[i]==1)):
                sub_labels = sub_labels.append(pd.Series([1], index=[i]))
            elif ((dataset_df[sens_attr[0]].iloc[i] == 1) & (dataset_df[decision_label].iloc[i]==0)):
                sub_labels = sub_labels.append(pd.Series([2], index=[i]))
            elif ((dataset_df[sens_attr[0]].iloc[i] == 1) & (dataset_df[decision_label].iloc[i]==1)):
                sub_labels = sub_labels.append(pd.Series([3], index=[i]))
    else:
        print("This function is made to work with 1 or 2 sensitive attributes.")
    sub_labels = sub_labels.tolist()
    return sub_labels


'-----------------------------------------------------------------------------'
#Function to prepare the dataset, obtain num and categ variables
def preprocess(df, sensitive_attr, y_label):
    
    #convert aif data to pandas dataframe
    if "aif360.datasets" in str(type(df)):
        orig = df.convert_to_dataframe()
        orig_df = orig[0]
    # or if the given dataset is already a dataframe
    else:
        orig_df = df
    
    # special treatment for categ. transform. if a numerical age column is used 
    # if ('age' in orig_df.columns) & (orig_df['age'].nunique()>2):
    #     for i in range(len(orig_df)):
    #         if orig_df['age'][i] >= 40:
    #             orig_df['age'][i] = 1
    #         else:
    #             orig_df['age'][i] = 0                
    # else:
    #     pass
        
    #identify numerical and categorical variables and adjust column types
    #NOTE: It is assumed that all categorical variables are binary.
            
    max_unique = 3
    num_idx = []
    cat_idx = []
    for i in range(len(orig_df.columns)):
        if orig_df.iloc[:,i].nunique() >= max_unique:
            orig_df.iloc[:,i] = orig_df.iloc[:,i].astype('float')
            num_idx.append(i)
        else:
            orig_df.iloc[:,i] = orig_df.iloc[:,i].astype('object')
            cat_idx.append(i)
        
    #add the sub-group labels
    sub_labels = sublabel(orig_df, sensitive_attr, y_label)
    orig_df['sub_labels'] = sub_labels
    orig_df['sub_labels'] = orig_df['sub_labels'].astype('object')     
        
    return orig_df, num_idx, cat_idx


'----------------------------------------------------------------------------'

#Categorical and numerical column index finder
def type_lists(df):
    #NOTE: It is assumed that all categorical variables are binary.
    num_idx = []
    cat_idx = []
    for i in range(len(df.columns)):
        if df.iloc[:,i].dtype == 'float64' or df.iloc[:,i].dtype == 'int64':
            num_idx.append(i)
        else:
            cat_idx.append(i)
    
    return num_idx, cat_idx


'----------------------------------------------------------------------------'

#Function for the feature scaling of only the numerical  features
#NOTE: Feature scaling is done separately on training and the test set in order
#not to leak any info. about the distribution from the test set.
def scale(X_train, X_test):
    
    scale = MinMaxScaler()
    #scaling TRAINING SET with the scaling object
    x_cat = X_train.select_dtypes(exclude=['float'])
    x_cat.reset_index(drop=True, inplace=True)
    
    x_num = X_train.select_dtypes(exclude='object')
    x_num = pd.DataFrame(scale.fit_transform(x_num))
    #concat back the numerical and categorial columns as training set
    X_train_scaled = pd.concat([x_num,x_cat], axis=1)
    
    #scaling TEST SET with the scaling object
    x_cat = X_test.select_dtypes(exclude=['float'])
    x_cat.reset_index(drop=True, inplace=True)
    
    x_num = X_test.select_dtypes(exclude='object')
    x_num = pd.DataFrame(scale.transform(x_num))
    #concat back the numerical and categorial columns as test set
    X_test_scaled = pd.concat([x_num,x_cat], axis=1)

    return X_train_scaled, X_test_scaled

'----------------------------------------------------------------------------'
def aif_dataset_metrics(X_train, y_train, prot_attr, favorable_lab, unfavorable_lab):
    #ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
    #find all the possible subgroups
    sens_subgroups = []
    for i in range(0,2):
        for k in range(0,2):
            sens_g = [{prot_attr[0]:k, prot_attr[1]:i}]
            sens_subgroups.append(sens_g)
            #print(sens_g)
    # print("Calculating the metrics for ", sens_subgroups)
    
    X_train_c = deepcopy(X_train)
    X_train_c['y_train'] = y_train
    aif_binary = BinaryLabelDataset(df=X_train_c, label_names=['y_train'], 
                                                    protected_attribute_names=prot_attr) 
    #assign the favorable and unfavorable labels to the datasets
    aif_binary.favorable_label = favorable_lab
    aif_binary.unfavorable_label = unfavorable_lab
    
    results_dict = {}
    #find all the possible combinations of the privileged and unprivileged subgroups
    #repeat this part in the final metrics calculations
    for i in range(0,3):
        for k in range(0,3):
            if (k+1+i)<=3:
                unprivileged_groups = sens_subgroups[(k)]
                privileged_groups = sens_subgroups[(k+1+i)]
                concat_name = str(unprivileged_groups)+str(privileged_groups)
            else:
                continue

            ind = []
            results = {"Disparate Impact Ratio": [], "Dem Parity Difference": [], 
                       "Consistency": []}
            #Construction 1
            #get the metrics that can be calculated on a single dataset
            metric_orig = BinaryLabelDatasetMetric(aif_binary, 
                                                        unprivileged_groups=unprivileged_groups,
                                                        privileged_groups=privileged_groups)
            #Disparate Impact ratio between privileged and unprivileged groups.
            di = metric_orig.disparate_impact()
            #Demographic parity difference between privileged and unprivileged groups.
            spd = metric_orig.statistical_parity_difference()
            #Individual Fairness: 1)Consistency, 2) Euclidean Distance between individuals.
            consistency = metric_orig.consistency()
            
            ind += ["Original Dataset Results"]
            results["Disparate Impact Ratio"].append(di)
            results["Dem Parity Difference"].append(spd)
            results["Consistency"].append(consistency[0])
            results_dict[concat_name] = results
            
    df_results = pd.DataFrame.from_dict(results_dict).T
    
    #find the aggregated result of privileged and unprivileged subgroups
    priv, unpriv = label_status(df_results, prot_attr) 
    aggregated_groups = concat_groups(X_train_c, priv, unpriv)
    prot_attrb = ['aggr_gr']
    priv_gr = [{'aggr_gr':1}]
    unpriv_gr = [{'aggr_gr':0}]
    aggregated_groups = aggregated_groups.drop([prot_attr[0], prot_attr[1]], axis=1) 
    #get the aggregated group metrics
    aggr_res_dict = {}
    indx = []
    aggr_results = {"Disparate Impact Ratio": [], "Dem Parity Difference": [], 
               "Consistency": []}
    concat_name = str(unpriv_gr)+str(priv_gr)
    aif_aggr = BinaryLabelDataset(df=aggregated_groups, label_names=['y_train'], 
                                                    protected_attribute_names=prot_attrb)
    aggr_metric = BinaryLabelDatasetMetric(aif_aggr, unprivileged_groups=unpriv_gr,
                                           privileged_groups=priv_gr)
    #Disparate Impact ratio between privileged and unprivileged groups.
    dia = aggr_metric.disparate_impact()
    #Demographic parity difference between privileged and unprivileged groups.
    spda = aggr_metric.statistical_parity_difference()
    #Individual Fairness: 1)Consistency, 2) Euclidean Distance between individuals.
    consistencya = aggr_metric.consistency()
    
    indx += ["Original Dataset Results"]
    aggr_results["Disparate Impact Ratio"].append(dia)
    aggr_results["Dem Parity Difference"].append(spda)
    aggr_results["Consistency"].append(consistencya[0])
    aggr_res_dict[concat_name] = aggr_results
    df_aggr = pd.DataFrame.from_dict(aggr_res_dict).T

    return df_results, df_aggr, priv, unpriv


'----------------------------------------------------------------------------'
# Function to decide the favorable and unfavorable labels in a metrics table
def label_status(metrics_table, sens_atr_list):
    #for the subgroup {sens_a1:1, sens_a2:0}
    unpriv_poss_10 = 0
    priv_poss_10 = 0
    #for the subgroup {sens_a1:0, sens_a2:1}
    unpriv_poss_01 = 0
    priv_poss_01 = 0
    
    #final lists of privileged and unprivileged groups (base groups are added)
    priv_list = [[{str(sens_atr_list[0]): 1, str(sens_atr_list[1]): 1}]]
    unpriv_list = [[{str(sens_atr_list[0]): 0, str(sens_atr_list[1]): 0}]]
    
    #conditions for the subgroup {sens_a1:1, sens_a2:0}
    #comparison with {sens_a1:0, sens_a2:0} unpriv
    if (metrics_table['Disparate Impact Ratio'][0][0] >= 0.8) & (metrics_table['Disparate Impact Ratio'][0][0] <= 1.2):
        unpriv_poss_10 += 1
    else:
        priv_poss_10 += 1
    #comparison with {sens_a1:1, sens_a2:1} priv
    if (metrics_table['Disparate Impact Ratio'][4][0] >= 0.8) & (metrics_table['Disparate Impact Ratio'][0][0] <= 1.2):
        priv_poss_10 += 1
    else:
        unpriv_poss_10 += 1
    #comparison with {sens_a1:0, sens_a2:1} priv
    # if metrics_table['Disparate Impact Ratio'][1] >= 1.0:
    #     priv_poss_10 += 1
    # else:
    #     unpriv_poss_10 += 1
    
    #now assign the subgroup  {sens_a1:1, sens_a2:0} to a group priv/unpriv
    if priv_poss_10 == 2:
        priv_list.append([{str(sens_atr_list[0]): 1, str(sens_atr_list[1]): 0}])
    elif priv_poss_10 == 0:
        unpriv_list.append([{str(sens_atr_list[0]): 1, str(sens_atr_list[1]): 0}])
    elif priv_poss_10 == 1:
        if metrics_table['Disparate Impact Ratio'][0][0] < metrics_table['Disparate Impact Ratio'][4][0]:
            priv_list.append([{str(sens_atr_list[0]): 1, str(sens_atr_list[1]): 0}])
        else:
            unpriv_list.append([{str(sens_atr_list[0]): 1, str(sens_atr_list[1]): 0}])
    else:
        print("There is a problem with the calculation, counter for privileged is ", priv_list)
        
    
    #conditions for the subgroup {sens_a1:0, sens_a2:1}
    #comparison with {sens_a1:0, sens_a2:0} unpriv
    if (metrics_table['Disparate Impact Ratio'][3][0] >= 0.8) & (metrics_table['Disparate Impact Ratio'][3][0] <= 1.2):
        unpriv_poss_01 += 1
    else:
        priv_poss_01 += 1
    #comparison with {sens_a1:1, sens_a2:1} priv
    if (metrics_table['Disparate Impact Ratio'][2][0] >= 0.8) & (metrics_table['Disparate Impact Ratio'][2][0] <= 1.2):
        priv_poss_01 += 1
    else:
        unpriv_poss_01 += 1
    #comparison with {sens_a1:0, sens_a2:1} priv
    # if metrics_table['Disparate Impact Ratio'][1] >= 1.0:
    #     priv_poss_01 += 1
    # else:
    #     unpriv_poss_01 += 1
    
    if priv_poss_01 == 2:
        priv_list.append([{str(sens_atr_list[0]): 0, str(sens_atr_list[1]): 1}])
    elif priv_poss_01 == 0:
        unpriv_list.append([{str(sens_atr_list[0]): 0, str(sens_atr_list[1]): 1}])
    elif priv_poss_01 == 1:
        if metrics_table['Disparate Impact Ratio'][3][0] < metrics_table['Disparate Impact Ratio'][2][0]:
            priv_list.append([{str(sens_atr_list[0]): 0, str(sens_atr_list[1]): 1}])
        else:
            unpriv_list.append([{str(sens_atr_list[0]): 0, str(sens_atr_list[1]): 1}])
    else:
        print("There is a problem with the calculation, counter for privileged is ", priv_list)
        
    #report the bias amount based on the main priv and unpriv subgroups
    if metrics_table['Disparate Impact Ratio'][5][0] < 0.8:
        print("There is a significant bias between the main privileged and unprivileged groups, which is under the legal threshold with the value ", metrics_table['Disparate Impact Ratio'][5][0])
    else:
        print("There is not a very significant bias between the main privileged and unprivileged groups, which is above the legal threshold with the value ", metrics_table['Disparate Impact Ratio'][5][0])
    
    print("  ")
    
    # report the relative position of the in-between subgroups compared with each other
    if metrics_table['Disparate Impact Ratio'][1][0]< 0.8:
        print("It seems that subgroup with labels 1,0 is significantly more unprivileged than the subgroup with labels 0,1 in order")
    elif metrics_table['Disparate Impact Ratio'][1][0] >= 1.2:
        print("It seems that subgroup with labels 1,0 is sinificantly more privileged than the subgroup with labels 0,1 in order")
    elif (metrics_table['Disparate Impact Ratio'][1][0] > 0.8) & (metrics_table['Disparate Impact Ratio'][1][0] <= 1.0):
        print("It seems that subgroup with labels 1,0 is slightly more unprivileged than the subgroup with labels 0,1 in order")
    elif (metrics_table['Disparate Impact Ratio'][1][0] > 1.0) & (metrics_table['Disparate Impact Ratio'][1][0] < 1.2):
        print("It seems that subgroup with labels 1,0 is slightly more privileged than the subgroup with labels 0,1 in order")   
        
    return priv_list, unpriv_list

#Function to aggregate the subgroups into 2 main privileged and unprivileged groups
def concat_groups(X_test, priv_gr, unpriv_gr):
    unpriv_vals = []
    unpriv_names = []
    for x in range(len(unpriv_gr)):
        for i in unpriv_gr[x][0].values():
            unpriv_vals.append(i)
        for k in unpriv_gr[x][0].keys():
            unpriv_names.append(k)
        
    priv_vals = []
    priv_names = []
    for x in range(len(priv_gr)):
        for i in priv_gr[x][0].values():
            priv_vals.append(i)
        for k in priv_gr[x][0].keys():
            priv_names.append(k)
    
    priv_df = pd.DataFrame()
    for i in range(len(priv_gr)+1):
        if i == 1:
            continue
        else:
            data = X_test.loc[(X_test[priv_names[i]] == priv_vals[i]) & (X_test[priv_names[i+1]] == priv_vals[i+1])]
            priv_df = priv_df.append(data)
      
    unpriv_df = pd.DataFrame()
    for i in range(len(unpriv_gr)+1):
        if i == 1:
            continue
        else:
            data = X_test.loc[(X_test[unpriv_names[i]] == unpriv_vals[i]) & (X_test[unpriv_names[i+1]] == unpriv_vals[i+1])]
            unpriv_df = unpriv_df.append(data)
            
    unpriv_df['aggr_gr'] = 0     
    priv_df['aggr_gr'] = 1
    aggr_df = pd.concat([priv_df, unpriv_df], axis = 0)
    #aggr_df = aggr_df.drop(['sub_labels'], axis=1)
    aggr_df = aggr_df.sort_index(ascending = True)
    
    return aggr_df


#function to get the ratios of each supgroup with their positive and negative outcomes
def subgroup_ratios(df, prot_attr_names):
    #collect the subgroup counts from the dataframe
    counts_df = df['sub_labels'].value_counts().rename_axis('unique_values').reset_index(name='counts')
    counts_df = counts_df.iloc[np.argsort(counts_df['unique_values'])]
    counts_df = counts_df.reset_index(drop=1)  
    
    #find all the possible subgroup names
    sens_subgroups = []
    for i in range(0,2):
        for k in range(0,2):
            sens_g = [{prot_attr_names[0]:k, prot_attr_names[1]:i}]
            sens_subgroups.append(sens_g)

    base_percentages = []
    pos_percentages = []
    neg_percentages = []
    groups = []
    counter = 0
    perc_df = pd.DataFrame()
    for k in range(len(counts_df)):
        #skip certain iterations to avoid choosing wrong subgroup combinations 
        if (k == 1) | (k==3) | (k==5) | (k==7):
            continue
        else:
            #subgroup name
            name = sens_subgroups[counter][0]
            #ratio calculations from the value counts
            perc1 = ((counts_df['counts'][k])+(counts_df['counts'][k+1]))/(sum(counts_df['counts']))
            perc2 = (counts_df['counts'][k+1])/(sum(counts_df['counts']))
            perc3 = (counts_df['counts'][k])/(sum(counts_df['counts']))
            base_percentages.append(perc1)
            pos_percentages.append(perc2)
            neg_percentages.append(perc3)
            groups.append(str(name))
            counter += 1
    perc_df = pd.DataFrame({'Subgroup': groups, 'Base Ratio': base_percentages,
                            'Positive Ratio': pos_percentages, 
                            'Negative Ratio': neg_percentages})
    perc_df = perc_df.set_index('Subgroup')
    perc_df.index.names = ['Index']
    return perc_df

'----------------------------------------------------------------------------'
#Function to create AIF360 compatible datasets and calculate the metrics
def aif_metrics(X_test_res, y_test, prot_attr, privileged_groups, 
                unprivileged_groups, favorable_lab, unfavorable_lab):
    #ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
    
    aif_binary_pred = BinaryLabelDataset(df=X_test_res, label_names=['y_pred'], 
                                            protected_attribute_names=prot_attr) 
    aif_binary_orig = aif_binary_pred.copy()
    aif_binary_orig.labels = y_test
    
    #assign the favorable and unfavorable labels to the datasets
    aif_binary_orig.favorable_label = favorable_lab
    aif_binary_pred.favorable_label = favorable_lab
    aif_binary_orig.unfavorable_label = unfavorable_lab
    aif_binary_pred.unfavorable_label = unfavorable_lab
    
    ind = []
    results = { "AEO Difference": [], "Disparate Impact Ratio": [], 
               "Dem Parity Difference": [], "Predictive Parity Difference": [], 
               "Consistency": [],  "Accuracy": [], "Balanced accuracy": [],  
               "F1-Score": [], "Precision (PPV)":[],"Recall (TPR)": [], 
               "Specificity (TNR)":[]}
    
    #Construction 1
    #to construct this metric function, the predicted labels should be united with the test fetures to make a new datas
    metric_pred_test = BinaryLabelDatasetMetric(aif_binary_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    #Construction 2
    #both original test dataset with actual labels and the test dataset combined with predicted class labels need to be given to this function
    classified_metric = ClassificationMetric(aif_binary_orig, aif_binary_pred, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    
    #Checking Equalized Odds: average odds differecence, which is the avg. of differences in FPR&TPR for privileged and unprivileged groups.
    aeo = classified_metric.average_odds_difference()
    
    #Disparate Impact ratio between privileged and unprivileged groups.
    di = metric_pred_test.disparate_impact()
    
    #Demographic parity difference between privileged and unprivileged groups.
    spd = metric_pred_test.statistical_parity_difference()
    
    #Predictive parity difference: PPV difference between privileged and unprivileged groups.
    ppd = classified_metric.positive_predictive_value(privileged=False) - classified_metric.positive_predictive_value(privileged=True)
    
    #Individual Fairness: 1)Consistency, 2) Euclidean Distance between individuals.
    consistency = metric_pred_test.consistency()
    
    TPR = classified_metric.true_positive_rate() #recall
    TNR = classified_metric.true_negative_rate() #specificity
    PPV = classified_metric.positive_predictive_value() #precision
    bal_acc = (TPR+TNR)/2 #balanced accuracy
    f1 = 2*((PPV*TPR)/(PPV+TPR)) #f1 score
    acc = classified_metric.accuracy() 
    cm = classified_metric.binary_confusion_matrix()
   
    ind += ["Baseline model"]
    results["AEO Difference"].append(aeo)
    results["Disparate Impact Ratio"].append(di)
    results["Dem Parity Difference"].append(spd)
    results["Predictive Parity Difference"].append(ppd)
    results["Consistency"].append(consistency[0])
    results["Accuracy"].append(acc)
    results["Balanced accuracy"].append(bal_acc)
    results["F1-Score"].append(f1)
    results["Precision (PPV)"].append(PPV)
    results["Recall (TPR)"].append(TPR)
    results["Specificity (TNR)"].append(TNR)

    return results, cm

'----------------------------------------------------------------------------'

#Function to output the baseline results of imbalanced datasets
def baseline_metrics(classifier, X_train, X_test, y_train, y_test, prot_attr, 
                     favorable_lab, unfavorable_lab): 
    #ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
    #remove the subgroup labels based on the original class labels
    X_train = X_train.loc[:, X_train.columns != 'sub_labels']
    X_test = X_test.loc[:, X_test.columns != 'sub_labels']  

    y_train = np.asarray(y_train).flatten()
    y_train = y_train.astype('float')
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    X_test_res = pd.DataFrame(X_test.copy())
    X_test_res['y_pred'] = y_pred
    
    #counts of each subgorups prediction amount in the preddicted test set
    y_label = 'y_pred'
    #calculate the subgroup labels based on the predicted class labels
    sub_labels = sublabel(X_test_res, prot_attr, y_label)
    X_test_res['sub_labels'] = sub_labels
    ratio_table = subgroup_ratios(X_test_res, prot_attr)
    X_test_res = X_test_res.drop(['sub_labels'], axis=1)
  
    #find all the possible subgroups
    sens_subgroups = []
    for i in range(0,2):
        for k in range(0,2):
            sens_g = [{prot_attr[0]:k, prot_attr[1]:i}]
            sens_subgroups.append(sens_g)
            #print(sens_g)
    # print("Calculating the metrics for ", sens_subgroups)
    
    results_dict = {}
    #find all the possible combinations of the privileged and unprivileged subgroup combinations
    #repeat this part in the final metrics calculations
    for i in range(0,3):
        for k in range(0,3):
            if (k+1+i)<=3:
                unprivileged_groups = sens_subgroups[(k)]
                privileged_groups = sens_subgroups[(k+1+i)]
                concat_name = str(unprivileged_groups)+str(privileged_groups)
            else:
                continue
            
            results, cm = aif_metrics(X_test_res, y_test, prot_attr, privileged_groups, 
                                  unprivileged_groups, favorable_lab, unfavorable_lab)              
            results_dict[concat_name] = results
            
    df_results = pd.DataFrame.from_dict(results_dict).T
    df_results = df_results.applymap(lambda x: x[0])
    
    #find the aggregated results of privileged and unprivileged subgroups (NOT NEEDED ANYMORE)
    # aggregated_groups = concat_groups(X_test_res, priv, unpriv)
    #you need priv and unpriv lists given as input if you activate this part of the code
    # prot_attrb = ['aggr_gr']
    # priv_gr = [{'aggr_gr':1}]
    # unpriv_gr = [{'aggr_gr':0}]
    # aggregated_groups = aggregated_groups.drop([prot_attr[0], prot_attr[1]], axis=1)
    # aggr_results, cm2 = aif_metrics(aggregated_groups, y_test, prot_attrb, priv_gr, 
    #                                unpriv_gr, favorable_lab, unfavorable_lab)
    # df_aggr = pd.DataFrame.from_dict(aggr_results)
    
    return df_results, cm, ratio_table


'----------------------------------------------------------------------------'
#Function for elbow method calculation for kprototypes
def kprot_elbow(i, k, X_train, cat_list):
    from kmodes.util.dissim import ng_dissim
    costs = []
    n_clusters = []
    clusters_assigned = []
    from tqdm import tqdm
    for i in tqdm(range(i, k)):
        try:
            cluster = KPrototypes(n_clusters=i, init='Cao', cat_dissim=matching_dissim)
            clusters = cluster.fit_predict(X_train, categorical=cat_list)
            costs.append(cluster.cost_)
            n_clusters.append(i)
            clusters_assigned.append(clusters)
        except:
            print(f"Can't cluster with {i} clusters")
           
    plt.scatter(x=n_clusters, y=costs)
    plt.plot(n_clusters, costs)
    plt.show()
    
'-----------------------------------------------------------------------------'
def fix_memberships(X_train_new, cluster_obj):
    existing_clust = {}

    #create a dataframe per cluster in a dictionary
    for h in range(len(X_train_new['cluster_labels'].unique())):
       existing_clust[h] = X_train_new.loc[X_train_new['cluster_labels']==h]
        
    #moving single samples to another nearest cluster and re-arranging clust dfs
    for i in range(len(existing_clust)):  
        moving_samples = []
        x_clust = existing_clust[i]
        
        #count the number of samples from each sub group
        subg_counts = x_clust['sub_labels'].value_counts()
        subg_counts = subg_counts.to_dict()
        # print(subg_counts)
        #if there is only 1 sample from a subgroup, get its subgroup id
        subg_list = []
        for j in subg_counts:
            if int(subg_counts[j]) == 1:
                subg_list.append(j)
            else:
                continue
        #find the corresponding row(s) to this sample(s), remove it from the 
        #current cluster and append it to a list
        if len(subg_list)>0:
            for k in range(len(subg_list)):
                subg_idx = subg_list[k]
                sample_to_move = x_clust[x_clust['sub_labels']==subg_idx]
                #Note: ~ means bitwise not, inversing boolean mask - Falses to Trues and Trues to Falses.
                x_clust = x_clust.loc[~(x_clust.sub_labels.isin(sample_to_move['sub_labels'])),:]
                existing_clust[i] = x_clust
                moving_samples.append(sample_to_move)
            costs_dict = {}
            for l in range(len(moving_samples)):
                centroids = {}
                for m in range(len(existing_clust)):
                    #skip the cluster that the row is deleted from
                    if (moving_samples[l]['cluster_labels'].item() == m):
                        continue
                    else:
                        subg_count = existing_clust[m]['sub_labels'].value_counts()
                        subg_count = subg_count.to_dict()
                        
                        #get the centroid of the cluster if it has at least 1 sample
                        #from the same subgroup as the deleted row
                        for key in subg_count.keys():
                            if key == moving_samples[l]['sub_labels'].item():
                                centroids[m] = cluster_obj.cluster_centroids_[m]
                            else:
                                continue
                                                    
                #eliminate unwanted cols and find var. indexes for cost calculation
                sample = moving_samples[l].drop(['cluster_labels','sub_labels', 'class_labels'], axis=1)
                num_list, cat_list = type_lists(sample)
            
                #calculate all the costs (distances) between eligible centroids
                #and the deleted row
                sample_cost = {}
                # Numerical cost = sum of Euclidean distances
                Xnum, Xcat = split_num_cat(sample, cat_list)
                for n in centroids:
                    num_costs = euclidean_dissim(centroids[n][num_list].reshape(1,-1), 
                                           Xnum[0])
                    cat_costs = matching_dissim(centroids[n][cat_list].reshape(1,-1), 
                                           Xcat[0], X=Xcat, membship=None)
                    # Gamma relates the categorical cost to the numerical cost.
                    tot_costs = num_costs + 0.5 * cat_costs
                    sample_cost[n]= tot_costs
                #find the cluster that has the min distance to the point
                costs_dict[l] = sample_cost
                min_dist = min(sample_cost.values())
                min_clust = [key for key in sample_cost if sample_cost[key] == min_dist]
                
                #add the row to the cluster dataframe with the minimum cost
                row_to_append = moving_samples[l]
                existing_clust[min_clust[0]] = pd.concat([existing_clust[min_clust[0]],
                                                        row_to_append], axis=0)
        else:
            pass
        
    # Drop the cluster label column from all dataframes    
    for x in range(len(existing_clust)):
        existing_clust[x] = existing_clust[x].drop(['cluster_labels'], axis=1)
        
    return existing_clust


'-----------------------------------------------------------------------------'

def fix_memberships_fcm(X_train_new, X_famd, clust_centroids):
    existing_clust = {}

    #create a dataframe per cluster in a dictionary
    for h in range(len(X_train_new['cluster_labels'].unique())):
       existing_clust[h] = X_train_new.loc[X_train_new['cluster_labels']==h]
        
    #moving single samples to another nearest cluster and re-arranging clust dfs
    for i in range(len(existing_clust)):  
        moving_samples = []
        x_clust = existing_clust[i]
        
        #count the number of samples from each sub group
        subg_counts = x_clust['sub_labels'].value_counts()
        subg_counts = subg_counts.to_dict()
        # print(subg_counts)
        #if there is only 1 sample from a subgroup, get its subgroup id
        subg_list = []
        for j in subg_counts:
            if int(subg_counts[j]) == 1:
                subg_list.append(j)
            else:
                continue
        #find the corresponding row(s) to this sample(s), remove it from the 
        #current cluster and append it to a list
        if len(subg_list)>0:
            idx_to_move = []
            for k in range(len(subg_list)):
                subg_idx = subg_list[k]
                sample_to_move = x_clust[x_clust['sub_labels']==subg_idx]
                sp_to_move= sample_to_move.squeeze()
                sample_idx = (X_train_new == sp_to_move).all(axis=1).idxmax()
                idx_to_move.append(sample_idx)
                #I NEED TO FIND THE ROW ID OF THESE SAMPLES TO USE FOR FUZZY
                
                
                #Note: ~ means bitwise not, inversing boolean mask - Falses to Trues and Trues to Falses.
                x_clust = x_clust.loc[~(x_clust.sub_labels.isin(sample_to_move['sub_labels'])),:]
                existing_clust[i] = x_clust
                moving_samples.append(sample_to_move)
            costs_dict = {}
            for l in range(len(moving_samples)):
                centroids = {}
                for m in range(len(existing_clust)):
                    #skip the cluster that the row is deleted from
                    if (moving_samples[l]['cluster_labels'].item() == m):
                        continue
                    else:
                        subg_count = existing_clust[m]['sub_labels'].value_counts()
                        subg_count = subg_count.to_dict()
                        
                        #get the centroid of the cluster if it has at least 1 sample
                        #from the same subgroup as the deleted row
                        for key in subg_count.keys():
                            if key == moving_samples[l]['sub_labels'].item():
                                centroids[m] = clust_centroids[m]
                            else:
                                continue
                                                    
                #eliminate unwanted cols and find var. indexes for cost calculation
                sample = moving_samples[l].drop(['cluster_labels','sub_labels', 'class_labels'], axis=1)
                num_list, cat_list = type_lists(sample)
            
                #calculate all the costs (distances) between eligible centroids
                #and the deleted row
                sample_cost = {}
                # Numerical cost = sum of Euclidean distances
                for i in centroids:
                    num_costs = euclidean_dissim(centroids[i].reshape(1,-1), 
                                           np.asarray(X_famd.iloc[int(idx_to_move[l]),:]))
                    sample_cost[i]= num_costs
                #find the cluster that has the min distance to the point
                costs_dict[l] = sample_cost
                min_dist = min(sample_cost.values())
                min_clust = [key for key in sample_cost if sample_cost[key] == min_dist]
                
                #add the row to the cluster dataframe with the minimum cost
                row_to_append = moving_samples[l]
                existing_clust[min_clust[0]] = pd.concat([existing_clust[min_clust[0]],
                                                        row_to_append], axis=0)
        else:
            pass
        
    # Drop the cluster label column from all dataframes    
    for x in range(len(existing_clust)):
        existing_clust[x] = existing_clust[x].drop(['cluster_labels'], axis=1)
        
    return existing_clust

'-----------------------------------------------------------------------------'
#Function that oversamples and classifies each cluster after applying the clustering algorithm
def oversample(existing_clust):
    oversampled_clust = {}
    sublabel_list = {}
    # Oversampling the re-arranged cluster dataframes
    for o in range(len(existing_clust)):     
        xd = existing_clust[o]
        # Clean cluster labels since it is the same value for all instances
        x = xd.drop(['sub_labels'], axis=1)
        y = existing_clust[o]['sub_labels']
        y = y.astype('int')
        if y.nunique() == 1:
            oversampled_clust[o] = xd
            continue
        else:
            num_l, cat_features = type_lists(x) 
            #sm = SMOTENC(cat_features, sampling_strategy='not majority', random_state=42, k_neighbors=1)
            sm = SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=1)
            #sm = ADASYN(sampling_strategy='not majority', random_state=42,n_neighbors=1)   
            x_res, y_res = sm.fit_resample(x, y)
            new_df = pd.concat([x_res, pd.DataFrame(y_res)], axis=1)
            oversampled_clust[o] = new_df
            
        sublabel_list[o] = y_res.unique()

    return oversampled_clust, sublabel_list


'-----------------------------------------------------------------------------'
#function to calculate classification models per cluster (for option 1&2)
def class_models(class_model, cluster_dict):
    lr_models = {}

    for i in cluster_dict:
        data_part = cluster_dict[i]
        data_part.reset_index(drop=True, inplace=True)   
        data_part = data_part.drop(['sub_labels'], axis=1)
        
        X_train = data_part.loc[:, data_part.columns != 'class_labels']
        y_train = data_part.loc[:, data_part.columns == 'class_labels']
        y_train = np.asarray(y_train).flatten()
        y_train = y_train.astype('float')
        model = class_model.fit(X_train, y_train) 
        lr_models[i] = deepcopy(model)
    #to check if models are unique   
    #for idx in range(len(lr_models)):
    #    print(idx, lr_models[idx].intercept_)   
    return lr_models


'-----------------------------------------------------------------------------'
#Option1 - using the whole oversampled training set

def predict_whole_set(model, cluster_dict, X_test):
    df = pd.DataFrame()
    
    for i in range(len(cluster_dict)):
        data_part = cluster_dict[i]
        df = pd.concat([df,data_part], axis=0)
        
    df.reset_index(drop=True, inplace=True)   
    df = df.drop(['sub_labels'], axis=1)
    X_train = df.loc[:, df.columns != 'class_labels']
    y_train = df.loc[:, df.columns == 'class_labels']
    y_train = np.asarray(y_train).flatten()
    y_train = y_train.astype('float')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    X_test_pred = pd.DataFrame(X_test.copy())
    X_test_pred['y_pred'] = y_pred
    
    return X_test_pred


'-----------------------------------------------------------------------------'
#Option2 - train model for each cluster, find each test sample's cluster to use correct model
# k-prototypes based class prediction
def predict_per_model(model, cluster_dict, cluster_object, X_test, costs, 
                      categorical_columns, subl_list, test_subls):
    #find the right cluster the each sample belongs to  
    lr_models = class_models(model, cluster_dict)
    y_preds = []
    for i in range(len(X_test)):        
        df_to_remove = []
        # print("sublabel id of the sample:", test_subls[i])
        for k in range(len(subl_list)):
            if test_subls[i] in subl_list[k]:
                pass
            else:
                df_to_remove.append(k)
        # print("removed clusters that doesn't have the sample subg':", 
        #       df_to_remove)
        
        if len(df_to_remove) == 0:
            index = np.where(costs[str(i)] == costs[str(i)].min())[0][0]
        else:
            #make the distance to the clusters that doesn't have the sublabel too great
            for j in range(len(df_to_remove)):
                costs[str(i)][df_to_remove[j]] = 1000000
            #new_clusts = [np.delete(costs[str(i)], df_to_remove)]
            index = np.where(costs[str(i)] == costs[str(i)].min())[0][0]
        # print("clusters to consider for class prediction:", costs[str(i)])
        # print("cluster id of the sample:", index)
        predicted = lr_models[index].predict(np.asarray(X_test.iloc[i,:]).reshape(1,-1))
        y_preds.append(predicted[0])
        
    X_test_pred = pd.DataFrame(X_test.copy())
    X_test_pred['y_pred'] = y_preds
    
    return X_test_pred


#for kmeoids with FAMD components
def kmed_dists(X, centroids):
    from scipy.spatial import distance
    n_points = X.shape[0]
    costs_dict = {}

    #labels = np.empty(n_points, dtype=np.uint16)
    for ipoint in range(n_points):
        sample_cost = []
        # Numerical cost = sum of Euclidean distances
        # Categorical cost = sum of matching dissimilarity
        for k in range(len(centroids)):
            costs = distance.euclidean(centroids[k], X.iloc[ipoint])
            sample_cost.append(costs)
        #reshape to see the distance of a point to each centroid in a row
        sample_cost = np.asarray(sample_cost).flatten() #.reshape(1,-1) to make it row
        costs_dict[ipoint] = sample_cost
                   
    return costs_dict


# for k-medoids based class prediction
def predict_w_clusters(model, cluster_dict, X_test, costs, subl_list, test_subls):
    
    lr_models = class_models(model, cluster_dict)
    y_preds = []
    for i in range(len(X_test)):
        df_to_remove = []
        # print("sublabel id of the sample:", test_subls[i])
        for k in range(len(subl_list)):
            if test_subls[i] in subl_list[k]:
                pass
            else:
                df_to_remove.append(k)
        # print("removed clusters that doesn't have the sample subg':", 
        #       df_to_remove)
        
        if len(df_to_remove) == 0:
            index = np.where(costs[i] == costs[i].min())[0][0]
        else:
            #make the distance to the clusters that doesn't have the sublabel too great
            for j in range(len(df_to_remove)):
                costs[i][df_to_remove[j]] = 1000000
            #new_clusts = [np.delete(costs[str(i)], df_to_remove)]
            index = np.where(costs[i] == costs[i].min())[0][0]
        # print("clusters to consider for class prediction:", costs[i])
        # print("cluster id of the sample:", index)
        predicted=lr_models[index].predict(np.asarray(X_test.iloc[i,:]).reshape(1,-1))
        y_preds.append(predicted[0])
        
    X_test_pred = pd.DataFrame(X_test.copy())
    X_test_pred['y_pred'] = y_preds
    
    return X_test_pred

 #Fuzzy c-means based class prediction
def predict_w_fuzzy(model, cluster_dict, X_test_orig, X_test_r, centroids, 
                    subl_list, test_subls):  
    lr_models = class_models(model, cluster_dict)
    y_preds = []
    costs = {}
    
    for ipoint in range(len(X_test_r)):
        sample_cost = []
        for i in centroids:
            s_cost = euclidean_dissim(np.asarray(X_test_r.iloc[ipoint,:]),
                                      i.reshape(1,-1)                                   )
            sample_cost.append(s_cost)
        #reshape to see the distance of a point to each centroid in a row
        sample_cost = np.asarray(sample_cost).flatten() #.reshape(1,-1) to make it row
        costs[str(ipoint)] = sample_cost 
    # print(costs)
    
    for i in range(len(X_test_orig)):
        df_to_remove = []
        # print("sublabel id of the sample:", test_subls.iloc[i])
        for k in range(len(subl_list)):
            if test_subls.iloc[i] in subl_list[k]:
                pass
            else:
                df_to_remove.append(k)
        # print("removed clusters that doesn't have the sample subg':", 
        #       df_to_remove)
        
        if len(df_to_remove) == 0:
            index = np.where(costs[str(i)] == costs[str(i)].min())[0][0]
        else:
            #make the distance to the clusters that doesn't have the sublabel too great
            for j in range(len(df_to_remove)):
                costs[str(i)][df_to_remove[j]] = 1000000

            index = np.where(costs[str(i)] == costs[str(i)].min())[0][0]
        # print("clusters to consider for class prediction:", costs[str(i)])
        # print("cluster id of the sample:", index)
        predicted=lr_models[index].predict(np.asarray(X_test_orig.iloc[i,:]).reshape(1,-1))
        y_preds.append(predicted[0])
        
    X_test_pred = pd.DataFrame(X_test_orig.copy())
    X_test_pred['y_pred'] = y_preds
    
    return X_test_pred

'-----------------------------------------------------------------------------'
#Option3 - giving weights to each calculated model per cluster

def split_num_cat(X, categorical):
    """Extract numerical and categorical columns.
    Convert to numpy arrays, if needed.

    :param X: Feature matrix
    :param categorical: Indices of categorical columns
    """
    Xnum = np.asanyarray(X.iloc[:, [ii for ii in range(X.shape[1])
                               if ii not in categorical]]).astype(np.float64)
    Xcat = np.asanyarray(X.iloc[:, categorical])
    Xcat, encmap = encode_features(Xcat)
    
    return Xnum, Xcat

# Cost (distance) calculation for the k-prototypes algorithm
def labels_cost(X, centroids, num_dissim, cat_dissim, gamma, membship=None):
    """Calculate labels and cost function given a matrix of points and
    a list of centroids for the k-prototypes algorithm.
    """
    numerical, categorical = type_lists(X)
    Xnum, Xcat = split_num_cat(X, categorical)
      
    n_points = Xnum.shape[0]
    Xnum = check_array(Xnum)
    costs_dict = {}

    #labels = np.empty(n_points, dtype=np.uint16)
    for ipoint in range(n_points):
        sample_cost = []
        # Numerical cost = sum of Euclidean distances
        # Categorical cost = sum of matching dissimilarity
        for k in range(len(centroids)):
            num_costs = num_dissim(centroids[k][numerical].reshape(1,-1), 
                                   Xnum[ipoint])
            cat_costs = cat_dissim(centroids[k][categorical].reshape(1,-1), 
                                   Xcat[ipoint], X=Xcat, membship=membship)
            # Gamma relates the categorical cost to the numerical cost.
            tot_costs = num_costs + gamma * cat_costs
            sample_cost.append(tot_costs)
        #reshape to see the distance of a point to each centroid in a row
        sample_cost = np.asarray(sample_cost).flatten() #.reshape(1,-1) to make it row
        costs_dict[str(ipoint)] = sample_cost
                   
    return costs_dict


# weighted class prediction function based on k-protoypes
def predict_w_weights(model, cluster_dict, costs, X_test, subl_list, test_subls):
    lr_models = class_models(model, cluster_dict)   
    weighted_preds = []
    
    #normalize the costs(distances) to the cluster centroids
    scaler = MinMaxScaler()
    
    for i in range(len(X_test)):
        #first eliminate the clusters from the distances list (cost) by assigning
        #1m that did not have the sublabel of the test label during the training
        clust_id_to_use = []
        # print("sublabel id of the sample:", test_subls[i])
        for k in range(len(subl_list)):
            if test_subls[i] in subl_list[k]:
                clust_id_to_use.append(k)
            else:
                pass
        # print("clusters that will be used for weighted prediction:",
        #       clust_id_to_use)
        
        #then calculate the normalized dists, avg dists and transformed dists
        weight = 0
        scaled_costs = scaler.fit_transform(costs[str(i)].reshape(-1,1))
        avg_d = np.mean(scaled_costs)
        new_costs = [(avg_d/(i+1)) for i in scaled_costs]
        threshold = sum(new_costs)/2
        # print("Threshold: ", threshold)

        for k in clust_id_to_use:
            weight += (new_costs[k])*(lr_models[k].predict(np.array(X_test.iloc[i,:]).reshape(1, -1)))
        # print("total weight:", weight)
        
        #decide the final class label
        if weight >= threshold:
            weight = 1
        else:
            weight = 0
        weighted_preds.append(weight)
    
    X_test_pred = pd.DataFrame(X_test.copy())
    X_test_pred['y_pred'] = weighted_preds
    
    return X_test_pred


# weighted class prediction function based on k-medoids
def predict_w_weights_kmed(model, cluster_dict, costs, X_test, subl_list, test_subls):
    #first create a dictionary from gower dists array
    # keys = (np.arange(len(costs)))
    # keys = keys.astype('str')
    # costs_dict = dict(zip(keys, costs))
    
    lr_models = class_models(model, cluster_dict)   
    weighted_preds = []
    
    #normalize the costs(distances) to the cluster centroids
    scaler = MinMaxScaler()
    
    for i in range(len(X_test)):        
        clust_id_to_use = []
        # print("sublabel id of the sample:", test_subls[i])
        #identify the cluster id's that are trained with the test sample's subgroup 
        for k in range(len(subl_list)):
            if test_subls[i] in subl_list[k]:
                clust_id_to_use.append(k)
            else:
                pass
        # print("clusters that will be used for weighted prediction:",
        #       clust_id_to_use)
        
        #find the weighted class prediction per sample
        weight = 0

        scaled_costs = scaler.fit_transform(costs[i].reshape(-1,1))
        avg_d = np.mean(scaled_costs)
        new_costs = [(avg_d/(i+1)) for i in scaled_costs]
        threshold = sum(new_costs)/2
        # print("Threshold: ", threshold)

        for k in clust_id_to_use:
            weight += (new_costs[k])*(lr_models[k].predict(np.array(X_test.iloc[i,:]).reshape(1, -1)))
        # print("total weight:", weight)
        
        if weight >= threshold:
            weight = 1
        else:
            weight = 0
        weighted_preds.append(weight)
    
    X_test_pred = pd.DataFrame(X_test.copy())
    X_test_pred['y_pred'] = weighted_preds
    
    return X_test_pred

# weighted class prediction function based on fuzzy c-means
def predict_w_weights_fuzzy(model, cluster_dict, prob_list, X_test, subl_list,
                            test_subls):
    lr_models = class_models(model, cluster_dict)   
    weighted_preds = []
    prob_list = prob_list.T 
    
    for i in range(len(X_test)):
        clust_id_to_use = []
        # print("sublabel id of the sample:", test_subls.iloc[i])
        #identify the cluster id's that are trained with the test sample's subgroup 
        for k in range(len(subl_list)):
            if test_subls.iloc[i] in subl_list[k]:
                clust_id_to_use.append(k)
            else:
                pass
        # print("clusters that will be used for weighted prediction:",
        #       clust_id_to_use)
        
        weight = 0     
        #weighting factor for the eligible clusters' probabilities
        eligible_probs = [prob_list[i][clust_id_to_use]]
        sums = np.sum(eligible_probs)
        # print("Sum of eligible probabilities:", sums)
                
        for k in clust_id_to_use:
            weight += (prob_list[i][k]/sums)*(lr_models[k].predict(np.array(X_test.iloc[i,:]).reshape(1, -1)))
        # print("total weight:", weight)
        
        if weight >= 0.5:
            weight = 1
        else:
            weight = 0
        # print(weight)
        weighted_preds.append(weight)
    
    X_test_pred = pd.DataFrame(X_test.copy())
    X_test_pred['y_pred'] = weighted_preds
    
    return X_test_pred


'-----------------------------------------------------------------------------'
#Function to calculate the fairness metrics for given dataset(s) and the classifier
def metrics_calculate(X_test, X_test_pred, y_test, prot_attr_names, favorable_l,
                      unfavorable_l):
    #AIF360's consistency score has a future warning. Check it in the next update
    from warnings import simplefilter
    # ignore all future warnings
    simplefilter(action='ignore', category=FutureWarning)
    
    #To transform dataframes back to aif360 dataset, use test set with sens attrs.
    X_test_c = pd.DataFrame(X_test.copy())
    #drop the original sublabels (based on original class labels)
    X_test_c = X_test_c.drop(['sub_labels'], axis=1)
    y_pred = X_test_pred['y_pred']
    X_test_c['y_pred'] = y_pred
    
    #get the new subgroup distribution in the predicted outcomes
    y_label = 'y_pred'
    #calculate the new sublabel based on predicted labels
    sub_labels = sublabel(X_test_c, prot_attr_names, y_label)
    X_test_c['sub_labels'] = sub_labels
    ratio_table = subgroup_ratios(X_test_c, prot_attr_names)
    X_test_c = X_test_c.drop(['sub_labels'], axis=1)
  
    
    #find all the possible subgroups
    sens_subgroups = []
    for i in range(0,2):
        for k in range(0,2):
            sens_g = [{prot_attr_names[0]:k, prot_attr_names[1]:i}]
            sens_subgroups.append(sens_g)
            #print(sens_g)
    # print("Calculating the metrics for ", sens_subgroups)
    
    results_dict = {}
    for i in range(0,3):
        for k in range(0,3):
            if (k+1+i)<=3:
                unprivileged_groups = sens_subgroups[(k)]
                privileged_groups = sens_subgroups[(k+1+i)]
                concat_name = str(unprivileged_groups)+str(privileged_groups)
            else:
                continue
            #the comfusion matrix outputs will be the same in all iterations
            results, cm = aif_metrics(X_test_c, y_test, prot_attr_names, privileged_groups, 
                                  unprivileged_groups, favorable_l, unfavorable_l)       
            results_dict[concat_name] = results
    df_results = pd.DataFrame.from_dict(results_dict).T
    df_results = df_results.applymap(lambda x: x[0])
    
    return df_results, cm, ratio_table


'----------------------------------------------------------------------------'
# Function to plot the clustering results using t-sne algorithm
# Control&adjust the tsne hyperparameters here
def plot_tsne(dataset, clusters, perplex, learning_r):
    from sklearn.manifold import TSNE
    tsne_model = TSNE(n_components=2, perplexity=perplex, verbose=1, random_state=0,
                     learning_rate=learning_r, n_iter=4000)
    tsne = tsne_model.fit_transform(dataset)
    tsne = pd.DataFrame(tsne)
    tsne['k'] = clusters
    
    for cluster in np.unique(clusters): # plot data by cluster
        plt.scatter(x=tsne.where(tsne['k']==cluster)[0],
                    y=tsne.where(tsne['k']==cluster)[1], label="c"+str(cluster))
        plt.legend()

        