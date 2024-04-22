import pandas as pd
import pathlib

from typing import List, Union


def load_data(file_name: str, labels: dict = None) -> pd.DataFrame:
    '''
    This command loads .csv files from the Kaggle database and creates a 
    pandas dataframe with labeled columns.

    Input:
        file_path (str): File path to data file to be loaded
        labels (dict): Dictionary of variable names to label the file with.

    Returns (DataFrame): A pandas dataframe with consistent labels.
    '''
    # Load data 
    file = pd.read_csv(pathlib.Path(__file__).parents[1] / 'data' / f'{file_name}')

    # Remove created values
    created_vars = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'hogar_total', 
                    'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding',
                    'SQBdependency', 'SQBmeaned', 'agesq']
    file.drop(created_vars, axis = 1, inplace = True)

    # Return
    return file
    
