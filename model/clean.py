import numpy as np
import pandas as pd
import pathlib

from typing import List


def load_data(file_name: str, labels: dict = None) -> pd.DataFrame:
    """
    This command loads .csv files from the Kaggle database and creates a
    pandas dataframe with labeled columns.

    Input:
        file_path (str): File path to data file to be loaded
        labels (dict): Dictionary of variable names to label the file with.

    Returns (DataFrame): A pandas dataframe with consistent labels.
    """
    # Load data
    file = pd.read_csv(pathlib.Path(__file__).parents[1] / "data" / f"{file_name}")

    # Remove created values
    created_vars = [
        "SQBescolari",
        "SQBage",
        "SQBhogar_total",
        "hogar_total",
        "SQBedjefe",
        "SQBhogar_nin",
        "SQBovercrowding",
        "SQBdependency",
        "SQBmeaned",
        "agesq",
    ]
    # Remove calculated vars
    calc_vars = [
        "edjefe",
        "edjefa",
        "r4h1",
        "r4h2",
        "r4h3",
        "r4m1",
        "r4m2",
        "r4m3",
        "r4t1",
        "r4t2",
        "overcrowding",
        "tamhog",
        "tamviv",
        "dependency",
        "meaneduc",
    ]
    # Remove 2nd pair of dummy vars
    dum_vars = ["area2", "male"]

    # Remove because no variation (97% = 'techozinc')
    techo_material = ["techozinc", "techoentrepiso", "techocane", "techootro"]

    # Drop all variables in place
    for vars in [created_vars, calc_vars, dum_vars, techo_material]:
        file.drop(vars, axis=1, inplace=True)

    # Return output
    return file


def clean_educ_cats(df: pd.DataFrame) -> pd.DataFrame:
    """
    This command cleans education variables.

    Input:
        df: A Dataframe with pandas values to complete.

    Returns: (DataFrame) A pandas dataframe for consistent labels.
    """
    # Load list of values to clean
    educ = [
        "instlevel1",
        "instlevel2",
        "instlevel3",
        "instlevel4",
        "instlevel5",
        "instlevel6",
        "instlevel7",
        "instlevel8",
        "instlevel9",
        "instlevel99",
    ]

    # Create education missing dummy
    df["instlevel99"] = 0
    df.loc[df[educ].sum(axis=1) == 0, "instlevel99"] = 1
    # Thenn cut to proportions
    educ_combined = pd.from_dummies(df[educ]).astype("category")
    educ_combined = educ_combined.iloc[:, 0].cat.rename_categories(
        {
            "instlevel1": "1_no_schooling",
            "instlevel2": "2_some_primary",
            "instlevel3": "3_complete_primary",
            "instlevel4": "4_some_secondary",
            "instlevel5": "5_complete_secondary",
            "instlevel6": "6_some_technical",
            "instlevel7": "7_complete_technical",
            "instlevel8": "8_complete_tertiary",
            "instlevel9": "9_complete_graduate",
            "instlevel99": "99_missing",
        }
    )
    df["educ"] = educ_combined

    # Combine variables than convert to dummies
    df["educ"] = np.where(
        df["educ"].isin(
            [
                "6_some_technical",
                "7_complete_technical",
                "9_complete_graduate",
                "8_complete_tertiary",
            ]
        ),
        "6_any_postsecondary",
        df["educ"],
    )
    df["educ"] = np.where(df["educ"].isin(["99_missing"]), "1_no_schooling", df["educ"])

    # Create output dummies
    df = pd.get_dummies(df, columns=["educ"], dtype="float")
    df.drop(educ, axis=1, inplace=True)
    return df


def clean_marital_cats(df: pd.DataFrame) -> pd.DataFrame:
    """
    This command cleans marital status variables.

    Input:
        df: A Dataframe with pandas values to complete.

    Returns: (DataFrame) A pandas dataframe for consistent labels.
    """
    # Load list of values to clean
    marital = [
        "estadocivil1",
        "estadocivil2",
        "estadocivil3",
        "estadocivil4",
        "estadocivil5",
        "estadocivil6",
        "estadocivil7",
    ]

    # Then cut to proportions
    marital_combined = pd.from_dummies(df[marital]).astype("category")
    marital_combined = marital_combined.iloc[:, 0].cat.rename_categories(
        {
            "estadocivil1": "1_child",
            "estadocivil2": "2_partnered",
            "estadocivil3": "3_married",
            "estadocivil4": "4_divorced",
            "estadocivil5": "5_separated",
            "estadocivil6": "6_widower",
            "estadocivil7": "7_single",
        }
    )
    df["marital"] = marital_combined

    # Combine variables than convert to dummies
    df["marital"] = np.where(
        df["marital"].isin(["4_divorced", "5_separated", "6_widower", "7_single"]),
        "4_separated",
        df["marital"],
    )

    # Create output dummies
    df = pd.get_dummies(df, columns=["marital"], dtype="float")
    df.drop(marital, axis=1, inplace=True)
    return df


def clean_hhh_rel_cats(df: pd.DataFrame) -> pd.DataFrame:
    """
    This command cleans relation to the household head variables.

    Input:
        df: A Dataframe with pandas values to complete.

    Returns: (DataFrame) A pandas dataframe for consistent labels.
    """
    # Load list of values to clean
    hhh_rel = [
        "parentesco1",
        "parentesco2",
        "parentesco3",
        "parentesco4",
        "parentesco5",
        "parentesco6",
        "parentesco7",
        "parentesco8",
        "parentesco9",
        "parentesco10",
        "parentesco11",
        "parentesco12",
    ]

    # Then cut to proportions
    hhh_rel_combined = pd.from_dummies(df[hhh_rel]).astype("category")
    hhh_rel_combined = hhh_rel_combined.iloc[:, 0].cat.rename_categories(
        {
            "parentesco1": "1_hhh",
            "parentesco2": "2_hhh_spouse",
            "parentesco3": "3_offspring",
            "parentesco4": "4_Stepson/daughter",
            "parentesco5": "5_Son/daughter in law",
            "parentesco6": "6_Grandson/daughter",
            "parentesco7": "7_Mother/father",
            "parentesco8": "8_Mother/father in law",
            "parentesco9": "9_Brother/sister",
            "parentesco10": "10_Brother/sister in law",
            "parentesco11": "11_Other family member",
            "parentesco12": "12_Other non-family",
        }
    )
    df["hhh_rel"] = hhh_rel_combined

    # Combine variables than convert to dummies
    df["hhh_rel"] = np.where(
        df["hhh_rel"].isin(
            [
                "4_Stepson/daughter",
                "5_Son/daughter in law",
                "6_Grandson/daughter",
                "7_Mother/father",
                "8_Mother/father in law",
                "9_Brother/sister",
                "10_Brother/sister in law",
                "11_Other family member",
                "12_Other non-family",
            ]
        ),
        "4_other",
        df["hhh_rel"],
    )

    # Create output dummies
    df = pd.get_dummies(df, columns=["hhh_rel"], dtype="float")
    df.drop(hhh_rel, axis=1, inplace=True)
    return df


def clean_pared_material_cats(df: pd.DataFrame) -> pd.DataFrame:
    """
    This command cleans the predominant material on the outside wall variables.

    Input:
        df: A Dataframe with pandas values to complete.

    Returns: (DataFrame) A pandas dataframe for consistent labels.
    """

    # Load list of values to clean
    pared_material = [
        "paredblolad",
        "paredzocalo",
        "paredpreb",
        "pareddes",
        "paredmad",
        "paredzinc",
        "paredfibras",
        "paredother",
    ]

    # Then cut to proportions
    pared_material_comb = pd.from_dummies(df[pared_material]).astype("category")
    pared_material_comb = pared_material_comb.iloc[:, 0].cat.rename_categories(
        {
            "paredblolad": "block or brick",
            "paredzocalo": "socket",
            "paredpreb": "prefabricated or cement",
            "pareddes": "waste material",
            "paredmad": "wood",
            "paredzinc": "zink",
            "paredfibras": "natural fibers",
            "paredother": "other",
        }
    )

    df["pared_material"] = pared_material_comb

    # Combine variables than convert to dummies
    df["pared_material"] = np.where(
        df["pared_material"].isin(["zink", "waste material", "natural fibers"]),
        "other",
        df["pared_material"],
    )

    # Create output dummies
    df = pd.get_dummies(df, columns=["pared_material"], dtype="float")
    df.drop(pared_material, axis=1, inplace=True)

    return df


def clean_piso_material_cats(df: pd.DataFrame) -> pd.DataFrame:
    """
    This command cleans predominant material on the floor.
    Input:
        df: A Dataframe with pandas values to complete.

    Returns: (DataFrame) A pandas dataframe for consistent labels.
    """

    # Load list of values to clean
    piso_material = [
        "pisomoscer",
        "pisocemento",
        "pisoother",
        "pisonatur",
        "pisonotiene",
        "pisomadera",
    ]

    # Then cut to proportions
    piso_material_combined = pd.from_dummies(df[piso_material]).astype("category")
    piso_material_combined = piso_material_combined.iloc[:, 0].cat.rename_categories(
        {
            "pisomoscer": "mosaic,  ceramic,  terrazo",
            "pisocemento": "cement",
            "pisoother": "other",
            "pisonatur": "natural material",
            "pisonotiene": "no floor",
            "pisomadera": "wood",
        }
    )

    df["piso_material"] = piso_material_combined

    # Combine variables than convert to dummies
    df["piso_material"] = np.where(
        df["piso_material"].isin(["natural material", "no floor"]),
        "other",
        df["piso_material"],
    )

    # Create output dummies
    df = pd.get_dummies(df, columns=["piso_material"], dtype="float")
    df.drop(piso_material, axis=1, inplace=True)

    return df


def clean_sanitario_cats(df: pd.DataFrame) -> pd.DataFrame:
    """
    This command cleans toilet connected to.
    Input:
        df: A Dataframe with pandas values to complete.

    Returns: (DataFrame) A pandas dataframe for consistent labels.
    """

    # Load list of values to clean
    sanitario = [
        "sanitario1",
        "sanitario2",
        "sanitario3",
        "sanitario5",
        "sanitario6",
    ]

    # Then cut to proportions
    sanitario_combined = pd.from_dummies(df[sanitario]).astype("category")
    sanitario_combined = sanitario_combined.iloc[:, 0].cat.rename_categories(
        {
            "sanitario1": "1 no toilet",
            "sanitario2": "2 sewer or cesspool",
            "sanitario3": "3 septic tank",
            "sanitario5": "5 black hole or letrine",
            "sanitario6": "6 other",
        }
    )

    df["sanitario"] = sanitario_combined

    # Combine variables than convert to dummies
    df["sanitario"] = np.where(
        df["sanitario"].isin(["5 black hole or letrinel", "1 no toilet"]),
        "6 other",
        df["sanitario"],
    )

    # Create output dummies
    df = pd.get_dummies(df, columns=["sanitario"], dtype="float")
    df.drop(sanitario, axis=1, inplace=True)

    return df


def clean_tipovivi_cats(df: pd.DataFrame) -> pd.DataFrame:
    """
    This command cleans if the housing is fully paid for.
    Input:
        df: A Dataframe with pandas values to complete.

    Returns: (DataFrame) A pandas dataframe for consistent labels.
    """

    # Load list of values to clean
    tipovivi = ["tipovivi1", "tipovivi2", "tipovivi3", "tipovivi4", "tipovivi5"]

    # Then cut to proportions
    tipoviv_combined = pd.from_dummies(df[tipovivi]).astype("category")
    tipoviv_combined = tipoviv_combined.iloc[:, 0].cat.rename_categories(
        {
            "tipovivi1": "fully paid",
            "tipovivi2": "own",
            "tipovivi3": "rented",
            "tipovivi4": "precarious",
            "tipovivi5": "other",
        }
    )

    df["tipovivi"] = tipoviv_combined

    # Combine variables than convert to dummies
    df["tipovivi"] = np.where(
        df["tipovivi"].isin(["precarious"]), "other", df["tipovivi"]
    )

    # Create output dummies
    df = pd.get_dummies(df, columns=["tipovivi"], dtype="float")
    df.drop(tipovivi, axis=1, inplace=True)

    return df


def clean_sanitario_cats(df: pd.DataFrame) -> pd.DataFrame:
    """
    This command cleans if rubbish disposal mainly by.
    Input:
        df: A Dataframe with pandas values to complete.

    Returns: (DataFrame) A pandas dataframe for consistent labels.
    """

    # Load list of values to clean
    rubbish_disposal = [
        "elimbasu1",
        "elimbasu2",
        "elimbasu3",
        "elimbasu4",
        "elimbasu5",
        "elimbasu6",
    ]

    # Then cut to proportions
    rubbish_disposal_combined = pd.from_dummies(df[rubbish_disposal]).astype("category")
    rubbish_disposal_combined = rubbish_disposal_combined.iloc[
        :, 0
    ].cat.rename_categories(
        {
            "elimbasu1": "1 tanker truck",
            "elimbasu2": "2 botan hollow or buried",
            "elimbasu3": "3 burning",
            "elimbasu4": "4 throwing in an unoccupied space",
            "elimbasu5": " 5 throwing in river,  creek or sea",
            "elimbasu6": "6 other",
        }
    )

    df["rubbish_disposal"] = rubbish_disposal_combined

    # Combine variables than convert to dummies
    df["rubbish_disposal"] = np.where(
        df["rubbish_disposal"].isin(
            ["2 botan hollow or buried", "4 throwing in an unoccupied space"]
        ),
        "6 other",
        df["rubbish_disposal"],
    )

    # Create output dummies
    df = pd.get_dummies(df, columns=["rubbish_disposal"], dtype="float")
    df.drop(rubbish_disposal, axis=1, inplace=True)

    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values for variables with missing values:

    Input:
        df: A Dataframe with missing values

    Returns: (DataFrame) A pandas dataframe for consistent labels.
    """
    # change all nan in (v18q1, number of tablets household owns) to 0
    df["v18q1"] = df["v18q1"].fillna(0)
    df.drop("v18q", axis=1, inplace=True)

    # Replace NaN with 0 in 'v2a1'(Monthly rent) where 'tipovivi1'(fully paid) = 1.
    df.loc[df['tipovivi1'] == 1, 'v2a1'] = df.loc[df["tipovivi1"] == 1, "v2a1"].fillna(0)
    
    # Replace NaN values with 0 for other and precarious too
    df.loc[df['tipovivi4'] == 1, 'v2a1'] = df.loc[df["tipovivi4"] == 1, "v2a1"].fillna(0)
    df.loc[df['tipovivi5'] == 1, 'v2a1'] = df.loc[df["tipovivi5"] == 1, "v2a1"].fillna(0)

    return df


def collapse_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the dataframe to the household-level and choose an observation
    that is the most "senior" member of the household with a response.

    Input:
        df (pd.Dataframe): Individual-level dataframe

    Returns: (pd.Dataframe) Household-level dataframe identified by 'idhogar'
    """
    # Only select HHH rows that are valid heads
    parentesco1_eq_1 = df[df["hhh_rel_1_hhh"] == 1]  # 'parentesco1' = 1 (household head)
    parentesco2_eq_1 = df[df["hhh_rel_2_hhh_spouse"] == 1]  # 'parentesco2' = 1 (spouse/partner)
    parentesco3_eq_1 = df[df["hhh_rel_3_offspring"] == 1]  # 'parentesco3' = 1 (son/doughter)

    # Concatenate the filtered DataFrames to create a subset of unique 'idhogar' values
    df = pd.concat(
        [parentesco1_eq_1, parentesco2_eq_1, parentesco3_eq_1]
    ).drop_duplicates(subset="idhogar")

    # Drop extraneous HHH rel questions:
    df = df.loc[:, ~df.columns.str.startswith('hhh_rel_')]
    
    # ID drops for split
    df.drop('Id', axis = 1, inplace = True)
    df.drop('idhogar', axis = 1, inplace = True)

    return df

def drop_indiv_vars(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Remove remaining individual-level variables.

    Input:
        - df (pd.DataFrame): Household-level dataset

    Returns (df): Household-level dataframe with dropped variables.
    '''
    vars_indiv = ['rez_esc']
    df.drop(vars_indiv, axis = 1, inplace = True)
    return df
