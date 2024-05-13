import numpy as np
import pandas as pd
import pathlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from typing import List, Union


## Tune variable
def tune_sklearn_models(
        X_train: np.ndarray, y_train: np.ndarray, model: str
    ) -> Union[List, List, int]:
    '''
    Function to handle training and returning various models as part of the 
    pipeline.

    Input:
        X_train - ndarray of training data
        y_train - ndarray of training data
        X_val - ndarray of validation data
        y_val - ndarray of validation data
    '''
    # Establish model parameters
    if model == 'RandomForestClassifier':
        # 2 * 2 * 4 * 3 * 3 = 154 models
        params = {
            'n_estimators': [100, 250], # Num trees
            'criterion': ['gini', 'entropy'], # Loss options, not including 'log_loss' 
            'max_depth': [None, X_train.shape[1]/2, X_train.shape[1]/4, 10], # Depth of tree
            'min_samples_leaf': [1, 5, 10], # Num obs in leaf required
            'max_samples': [0.25, 0.50, 0.75] # Percent of sample to include
        }
        object = RandomForestClassifier()
    if model == "LogisticRegression":
        # 3 * 5 + 1 * 5 = 20 models
        params = {
            'penalty': ['l1', 'l2', 'elasticnet'], 
            'C' : [1.0, 0.1, 0.01, 0.001, 0.0001, 0.0001], # Lambda weight, inverse so smaller = larger penalty
            'l1_ratio': [0.10, 0.25, 0.50, 0.75, 0.90]
        }
        object = LogisticRegression()
    if model == "KNeighborsClassifier":
        # 5 = 5 models
        params = {
            'leaf_size': [5, 10, 20, 30, 50]
        }
        object = KNeighborsClassifier()

    # Tune model and extrat relevant output
    search = GridSearchCV(object, params, scoring = 'f1', cv = 5)
    search.fit(X_train, y_train)

    # Extract results
    params = search.cv_results_['params']
    test_score = search.cv_results_['mean_test_score']
    best = search.cv_results_['best'].index(1) # Extract the index of the best model

    # Return tuple of results
    return (params, test_score, best)

def tune_nn_model(
        X_train: np.ndarray, y_train: np.ndarray
    ) -> Union[List, List, int]:
    pass
    # TODO: Figure out neural net format.

# Determine final model
def visualize_acc():
    # TODO: Figure out how to compare - False pos / false neg
    pass

## Create a model object 
    # From final

# Prediction thresholding
def show_roc_auc():
    pass
## ROC / AUC to determine thresholds