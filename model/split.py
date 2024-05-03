import numpy as np
import pandas as pd
import sys as sys
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import List, Union


def cr_pmt_split(df: pd.DataFrame, label: str = 'Target', seed: int = 42):
    '''
    Take in the input data and split it into 3 sets of training, validation,
    tests split on the label and input variables. It uses a 70%, 10%, and 20%
    size split respectively.

    We resample using SMOTE for the training and validation sets and then the
    test set separately.

    Note that data should be cut to the household--level beforehand.

    Input:
        df (DataFrame): Input data to be split
        
    Returns:
        Test
    '''
    X = df.drop(columns = [label])  
    y = df[label]

    # Split data into temporary (80%) and test (20%) sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.2, \
                                                      random_state = seed)

    # Apply SMOTE to the training set
    smote = SMOTE()
    X_temp_resampled, y_temp_resampled = smote.fit_resample(X_temp, y_temp)
    X_test_resampled, y_test_resampled = smote.fit_resample(X_test, y_test)

    # Split the temporary set into training (70%) and validation (10%) sets
    X_train_resampled, X_val_resampled, y_train_resampled, y_val_resampled = \
        train_test_split(X_temp_resampled, y_temp_resampled, \
                         test_size= 1/8, random_state = seed)

    print("Training set size after SMOTE:", len(X_train_resampled))
    print("Validation set size:", len(X_val_resampled))
    print("Test set size:", len(X_test_resampled))

    # Return
    return X_train_resampled, y_train_resampled, \
        X_val_resampled, y_val_resampled, \
        X_test_resampled, y_test_resampled
