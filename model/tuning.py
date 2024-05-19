import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from joblib import dump

from typing import List, Union, Tuple

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
        # 3 * 1 * 4 * 4 * 3 = 144 models
        params = {
            'n_estimators': [25, 50, 100], #, 250], # Num trees
            'criterion': ['gini'], #, 'entropy'], # Loss options, not including 'log_loss' 
            'max_depth': [None, int(round(X_train.shape[1]/2)), int(round(X_train.shape[1]/4)), 5], # Depth of tree
            'min_samples_leaf': [1, 5, 10, 50], # Num obs in leaf required
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
        # 6 * 2 = 12 models
        params = {
            'n_neighbors': [3, 5, 10, 25, 50, 100],
            'weights': ['uniform', 'distance']
        }
        object = KNeighborsClassifier()

    # Tune model and extract relevant output, optimizing on F1 score
    search = GridSearchCV(object, params, scoring = 'f1_macro', cv = 5) # f1_Macro is unweighted average (fine b/c SMOTE)
    search.fit(X_train, y_train)

    # Extract results
    params = search.cv_results_['params']
    test_score = search.cv_results_['mean_test_score']
    best = list(search.cv_results_['rank_test_score']).index(1) # Extract the index of the best model

    # Get the best model
    best_model = search.best_estimator_

    # Perform prediction on each fold and store true and predicted labels
    predictions = []
    for train_idx, valid_idx in search.cv.split(X_train, y_train):
        X_fold_train, X_fold_valid = X_train[train_idx], X_train[valid_idx]
        y_fold_train, y_fold_valid = y_train[train_idx], y_train[valid_idx]
        model_fold = search.best_estimator_.set_params(**params[best])
        model_fold.fit(X_fold_train, y_fold_train)
        y_pred_fold = model_fold.predict(X_fold_valid)
        predictions.append((y_fold_valid, y_pred_fold))

    # Return tuple of results
    return (params, test_score, best, best_model, predictions)

#def tune_nn_model(
#        X_train: np.ndarray, y_train: np.ndarray
#    ) -> Union[List, List, int]:
#    pass
#    # TODO: Tensor flow + grid search, if time

# Determine final model
def visualize_acc(model, X_test, y_test):
    '''
    Function to visualize the accuracy and decision boundaries of the model.
    '''
    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy and F1 Score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score (macro): {f1}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def save_final_model(model, filename: str):
    '''
    Function to save the final model to disk.
    '''
    dump(model, filename)
    print(f"Model saved to {filename}")

# Prediction thresholding
def plot_roc_auc(model, X_test, y_test):
    '''
    Function to plot ROC curve and calculate AUC score.
    '''
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()