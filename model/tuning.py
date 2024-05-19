import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, auc, roc_curve
from joblib import dump

from typing import List, Union, Tuple

import matplotlib
import os

if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

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
    search = GridSearchCV(object, params, scoring = 'f1_macro', cv = 5, return_train_score = False) # f1_Macro is unweighted average (fine b/c SMOTE)
    search.fit(X_train, y_train)

    # Extract results
    params = search.cv_results_['params']
    test_score = search.cv_results_['mean_test_score']
    best = list(search.cv_results_['rank_test_score']).index(1) # Extract the index of the best model
    best_model = search.best_estimator_

    # Return tuple of results
    return (params, test_score, best, best_model)

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
    
    # Plot heatmap
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')

    # Add annotations
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, str(cm[i, j]), horizontalalignment='center', verticalalignment='center')

    plt.colorbar(label='Counts')
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
    Function to plot ROC curve and calculate AUC score for multiclass classification.
    '''
    # Binarize the labels
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_test_bin.shape[1]

    for i in range(n_classes):
        y_score = model.predict_proba(X_test)[:, i]
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 7))
    colors = ['blue', 'red', 'green', 'purple'] 
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve (class {i}) (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()