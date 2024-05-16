import numpy as np
import pandas as pd
import pathlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from typing import List, Union

#------------------------------------------------
# TODO:
#   - Visualizing model decision
#   - Accuracy evaluation 
#   - Save final model
# ------------------------------------------------

## Tune variable
def tune_sklearn_models(
        X_train: np.ndarray, y_train: np.ndarray, model: str
    ) -> Union[List, List, int]:
    '''
    Function to handle training and returning various models as part of the 
    pipeline. Currently handles:
        - Penalized multinomial regression
        - KNN
        - Random forests

    Input:
        X_train - ndarray of training data
        y_train - ndarray of training data
        model - Classifier object name in sklearn
    
    Output:
        (List, List, int):
            - parameters tested for a specific model
            - test accuracy for the CV
            - Which position in the lists perform best
    '''
    # Establish model parameters
    if model == 'RandomForestClassifier':
        # 2 * 2 * 4 * 3 * 3 = 154 models
        params = {
            'n_estimators': [25, 50, 100], #, 250], # Num trees
            'criterion': ['gini'], #, 'entropy'], # Loss options, not including 'log_loss' 
            'max_depth': [None, int(round(X_train.shape[1]/2)), int(round(X_train.shape[1]/4)), 5], # Depth of tree
            'min_samples_leaf': [1, 5, 10], # Num obs in leaf required
            'max_samples': [0.25, 0.50, 0.75] # Percent of sample to include
        }
        object = RandomForestClassifier()
    if model == "LogisticRegression":
        # 8 * 2 = 16 models
        params = {
            'penalty': ['l1', 'l2'], # No elastic net due to solver issues 
            'C' : [100, 10, 1.0, 0.1, 0.01, 0.001, 0.0001, 0.0001], # Lambda weight, inverse so smaller = larger penalty
        }
        object = LogisticRegression(max_iter = 100, solver = 'liblinear')
    if model == "KNeighborsClassifier":
        # 5 = 5 models
        params = {
            'n_neighbors': [3, 5, 10, 25, 50, 100]
        }
        object = KNeighborsClassifier()

    # Tune model and extract relevant output, optimizing on F1 score
    search = GridSearchCV(object, params, scoring = 'f1_macro', cv = 5) # f1_Macro is unweighted average (fine b/c SMOTE)
    search.fit(X_train, y_train)

    # Extract results
    params = search.cv_results_['params']
    test_score = search.cv_results_['mean_test_score']
    best = list(search.cv_results_['rank_test_score']).index(1) # Extract the index of the best model

    # Return tuple of results
    return (params, test_score, best)

#def tune_nn_model(
#        X_train: np.ndarray, y_train: np.ndarray
#    ) -> Union[List, List, int]:
#    pass
#    # TODO: Tensor flow + grid search, if time

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