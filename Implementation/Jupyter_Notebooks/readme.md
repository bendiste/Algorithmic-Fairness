For sampling using CTGAN, make sure first to create your virtual environement and install the (sdv, ctgan) along with the fairness requirements. 

Each dataset is fitted using the CTGAN and the classifiers (Random Forest, Logistic Regression and the Extreme Gradient Boosting), is all trained on the concatnated dataframe from both the training set and also the newly generated samples from the training set.

Run the notebooks and see the differences between each model SMOTE+Clustering Or the GANs. 
