import numpy as np
import pandas as pd
import pathlib

from typing import List, Union

## ----------------------------------------------------------------------------
# TODO for Cleaning commands:
#       - Write collapse vars for durable assets:
#       - Collpase command
## ----------------------------------------------------------------------------

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
    # Remove calculated vars
    calc_vars = ["edjefe", "edjefa", "r4h1", "r4h2", "r4h3", 
                 "r4m1", "r4m2", "r4m3", "r4t1", "r4t2", "overcrowding", 
                 "tamhog", "tamviv", "dependency", 
                 "meaneduc"]
    # Remove 2nd pair of dummy vars
    dum_vars = ["area2", "male"]
    
    # Drop all variables in place
    for vars in [created_vars, calc_vars, dum_vars]:
        file.drop(vars, axis = 1, inplace = True)

    # Return output
    return file
    

def clean_educ_cats(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This command cleans education variables.

    Input:
        df: A Dataframe with pandas values to complete.
    
    Returns: (DataFrame) A pandas dataframe for consistent labels.
    ''' 
    # Load list of values to clean
    educ = ['instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', \
        'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9', \
        'instlevel99']
    
    # Create education missing dummy
    df['instlevel99'] = 0
    df.loc[df[educ].sum(axis = 1) == 0, 'instlevel99'] = 1
    # Thenn cut to proportions
    educ_combined = pd.from_dummies(df[educ]).astype('category')
    educ_combined = educ_combined.iloc[:, 0].cat.rename_categories(
            {'instlevel1': '1_no_schooling', 
            'instlevel2': '2_some_primary', 
            'instlevel3': '3_complete_primary', 
            'instlevel4': '4_some_secondary', 
            'instlevel5': '5_complete_secondary', 
            'instlevel6': '6_some_technical',
            'instlevel7': '7_complete_technical', 
            'instlevel8': '8_complete_tertiary', 
            'instlevel9': '9_complete_graduate',
            'instlevel99' : '99_missing'}
        )
    df['educ'] = educ_combined

    # Combine variables than convert to dummies
    df['educ'] = np.where(
        df['educ'].isin(['6_some_technical', 
                         '7_complete_technical',
                         '9_complete_graduate', 
                         '8_complete_tertiary']
                        ), 
            '6_any_postsecondary', df['educ'])
    df['educ'] = np.where(
        df['educ'].isin(['99_missing']), 
            '1_no_schooling', df['educ'])
    df['educ'].value_counts(normalize = True)

    # Create output dummies
    df = pd.get_dummies(df, columns = ['educ'], dtype = 'float')
    df.drop(educ, axis = 1, inplace = True)
    return df


def clean_marital_cats(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This command cleans marital status variables.

    Input:
        df: A Dataframe with pandas values to complete.
    
    Returns: (DataFrame) A pandas dataframe for consistent labels.
    ''' 
    # Load list of values to clean
    marital = ['estadocivil1', 'estadocivil2', 'estadocivil3', 'estadocivil4', \
    'estadocivil5', 'estadocivil6', 'estadocivil7']
    
    # Then cut to proportions
    marital_combined = pd.from_dummies(df[marital]).astype('category')
    marital_combined = marital_combined.iloc[:, 0].cat.rename_categories(
            {'estadocivil1': '1_child', 
            'estadocivil2': '2_partnered', 
            'estadocivil3': '3_married', 
            'estadocivil4': '4_divorced', 
            'estadocivil5': '5_separated', 
            'estadocivil6': '6_widower', 
            'estadocivil7': '7_single'})
    df['marital'] = marital_combined

    # Combine variables than convert to dummies
    df['marital'] = np.where(
        df['marital'].isin(['4_divorced', 
                         '5_separated',
                         '6_widower', 
                         '7_single']), 
            '4_separated', df['marital'])
    df['marital'].value_counts(normalize = True)

    # Create output dummies
    df = pd.get_dummies(df, columns = ['marital'], dtype = 'float')
    df.drop(marital, axis = 1, inplace = True)
    return df


def clean_hhh_rel_cats(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This command cleans relation to the household head variables.

    Input:
        df: A Dataframe with pandas values to complete.
    
    Returns: (DataFrame) A pandas dataframe for consistent labels.
    ''' 
    # Load list of values to clean
    hhh_rel = ['parentesco1', 'parentesco2', 'parentesco3', 'parentesco4', \
            'parentesco5', 'parentesco6', 'parentesco7', 'parentesco8', \
            'parentesco9', 'parentesco10', 'parentesco11', 'parentesco12']
    
    # Then cut to proportions
    hhh_rel_combined = pd.from_dummies(df[hhh_rel]).astype('category')
    hhh_rel_combined = hhh_rel_combined.iloc[:, 0].cat.rename_categories(
        {'parentesco1': '1_hhh', 
         'parentesco2': '2_hhh_spouse', 
         'parentesco3': '3_offspring', 
         'parentesco4': '4_Stepson/daughter', 
         'parentesco5': '5_Son/daughter in law', 
         'parentesco6': '6_Grandson/daughter',
         'parentesco7': '7_Mother/father', 
         'parentesco8': '8_Mother/father in law', 
         'parentesco9': '9_Brother/sister',  
         'parentesco10': '10_Brother/sister in law',
         'parentesco11': '11_Other family member', 
         'parentesco12': '12_Other non-family'})
    df['hhh_rel'] = hhh_rel_combined

    # Combine variables than convert to dummies
    df['hhh_rel'] = np.where(
        df['hhh_rel'].isin(['4_Stepson/daughter', 
            '5_Son/daughter in law', 
            '6_Grandson/daughter',
            '7_Mother/father', 
            '8_Mother/father in law', 
            '9_Brother/sister',  
            '10_Brother/sister in law',
            '11_Other family member', 
            '12_Other non-family']), 
            '4_other', df['hhh_rel'])
    df['hhh_rel'].value_counts(normalize = True)

    # Create output dummies
    df = pd.get_dummies(df, columns = ['hhh_rel'], dtype = 'float')
    df.drop(hhh_rel, axis = 1, inplace = True)
    return df

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Handle missing values for variables with missing values:

    Input:
        df: A Dataframe with missing values
    
    Returns: (DataFrame) A pandas dataframe for consistent labels.
    '''
    
    #change all nan in (v18q1, number of tablets household owns) to 0
    df['v18q1'] = df['v18q1'].fillna(0)

    # Replace NaN values in 'v2a1'(Monthly rent payment) where 'tipovivi1'(fully paid) equals 1 with a 0.
    df.loc[df['tipovivi1'] == 1, 'v2a1'] = df.loc[df['tipovivi1'] == 1, 'v2a1'].fillna(0)

    return df

def collapse_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Collapse the dataframe to the household-level.
    '''
    # TODO: Need to write this.
    pass
