import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, auc, roc_curve, ConfusionMatrixDisplay
from joblib import dump

from typing import List, Union, Tuple

import matplotlib
import os

#if os.environ.get("DISPLAY", "") == "":
#    print("No display found. Using non-interactive Agg backend")
#    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# -------------------------------------
# Model Tuning
# -------------------------------------
def tune_sklearn_models(
    X_train: np.ndarray, y_train: np.ndarray, model: str
) -> Union[List, List, int]:
    """
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
        (List, List, int, model):
            - parameters tested for a specific model
            - test accuracy for the CV
            - Which position in the lists perform best
            - Classifier object
    """
    # Establish model parameters
    if model == "RandomForestClassifier":
        # 3 * 4 * 4 * 3 = 144 models
        params = {
            "n_estimators": [25, 50, 100], # Num trees
            "max_depth": [
                None,
                int(round(X_train.shape[1] / 2)),
                int(round(X_train.shape[1] / 4)),
                5,
            ],  # Depth of tree
            "min_samples_leaf": [1, 5, 10, 50],  # Num obs in leaf required
            "max_samples": [0.25, 0.50, 0.75],  # Percent of sample to include
        }
        new_params = {'randomforestclassifier__' + key: params[key] for key in params}
        object = make_pipeline(SMOTE(), RandomForestClassifier(criterion = 'gini'))
    if model == "LogisticRegression":
        # 8 * 2 = 16 models
        params = {
            "penalty": ["l1", "l2"],  # No elastic net due to solver issues
            "C": [
                100,
                10,
                1.0,
                0.1,
                0.01,
                0.001,
                0.0001,
                0.0001,
            ],  # Lambda weight, inverse so smaller = larger penalty
        }
        new_params = {'logisticregression__' + key: params[key] for key in params}
        object = make_pipeline(SMOTE(), LogisticRegression(max_iter=100, solver="liblinear"))
    if model == "KNeighborsClassifier":
        # 6 * 2 = 12 models
        params = {
            "n_neighbors": [3, 5, 10, 25, 50, 100],
            "weights": ["uniform", "distance"],
        }
        new_params = {'kneighborsclassifier__' + key: params[key] for key in params}
        object = make_pipeline(SMOTE(), KNeighborsClassifier())
                               
    # Tune model and extract relevant output, optimizing on F1 score
    search = GridSearchCV(
        object, 
        new_params, 
        scoring = "f1_macro", 
        cv = 5, 
        refit = True
    )  # f1_Macro is unweighted average (fine b/c SMOTE)
    search.fit(X_train, y_train)

    # Extract results
    params = search.cv_results_["params"]
    test_score = search.cv_results_["mean_test_score"]
    best = list(search.cv_results_["rank_test_score"]).index(1)

    # Return tuple of results
    return (params, test_score, best, search)

# -------------------------------------
# Model evaluation
# -------------------------------------
def visualize_acc(model, model_name: str, X_test: pd.DataFrame, y_test: pd.DataFrame, normalize='true'):
    """
    Function to visualize the accuracy and decision boundaries of the model.

    Input:
        - model (sklearn.classifier): Classifier model to show the accuracy
            of the model
        - X_test (pd.DataFrame): Testing data
        - y_test (pd.DataFrame): Array of training data

    Returns: Displays plotting
    """
    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy and F1 Score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score (macro): {f1}")

    # Confusion matrix:
    labels = ['Extreme\nPoverty', 
              'Moderate\nPoverty', 
              'Vulnerable\nHouseholds', 
              'Non-vulnerable\nHouseholds']
    cm = confusion_matrix(y_test, y_pred, normalize=normalize)
    
    # Display confusion matrix in-line with sci-kit learn formatting
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=labels)
    disp.plot()
    plt.title(f'Confusion Matrix - {model_name}') 
    plt.show()


# Prediction thresholding
def plot_roc_auc(model, model_name: str, X_test: pd.DataFrame, y_test: pd.DataFrame):
    """
    Function to plot ROC curve and calculate AUC score for multiclass 
    classification.

    Input:
        - model (sklearn.classifier): Classifier model to show the accuracy
            of the model
        - X_test (pd.DataFrame): Testing data
        - y_test (pd.DataFrame): Array of training data

    Returns: Displays plotting
    """
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
    colors = ["blue", "red", "green", "purple"]
    labels = ['Extreme Poverty', 
              'Moderate Poverty', 
              'Vulnerable Households', 
              'Non-vulnerable Households']
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"ROC curve ({labels[i]}) (area = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic - {model_name}")
    plt.legend(loc="lower right")
    plt.show()


# -------------------------------------
#  Helper functions
# -------------------------------------
def create_estimate_table(output: Union[list, np.ndarray, int], 
                          type: str, 
                          filename: str) -> None:
    '''
    Helper funciton to save the training values from the cross-validation
    into a CSV file to estimate the output.

    Input:
        output (Tuple): Tuple from our training function
        type (str): Type of model being added to a CSV
        filename (str): Filename

    Returns: None (saves results to disk)
    '''
    # Load list of dicts and array into the same format
    df = pd.DataFrame(output[0])
    df['f1_macro'] = output[1]
    df['type'] = type
    # Save output
    df.to_csv(filename)


def save_final_model(model, filename: str):
    """
    Function to save the final model to disk.

    Input:

    """
    dump(model, filename)
    print(f"Model saved to {filename}")
