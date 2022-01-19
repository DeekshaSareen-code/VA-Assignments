import collections
from pathlib import Path
from typing import Union, Optional
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.c_data_cleaning import *
from assignments.assignment1.a_load_file import read_dataset




##############################################
# Example(s). Read the comments in the following method(s)
##############################################


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def generate_label_encoder(df_column: pd.Series) -> LabelEncoder:
    """
    This method should generate a (sklearn version of a) label encoder of a categorical column
    :param df_column: Dataset's column
    :return: A label encoder of the column
    """
    from sklearn.preprocessing import LabelEncoder

    labelling = LabelEncoder()
    return labelling.fit(df_column)



def generate_one_hot_encoder(df_column: pd.Series) -> OneHotEncoder:
    """
    This method should generate a (sklearn version of a) one hot encoder of a categorical column
    :param df_column: Dataset's column
    :return: A label encoder of the column
    """
    from sklearn.preprocessing import OneHotEncoder

    hone = OneHotEncoder()
    hone.fit(df_column.values.reshape(-1, 1))
    return hone


def replace_with_label_encoder(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    """
    This method should replace the column of df with the label encoder's version of the column
    :param df: Dataset
    :param column: column to be replaced
    :param le: the label encoder to be used to replace the column
    :return: The df with the column replaced with the one from label encoder
    """
    df2 = df.copy()
    df2[column] = le.transform(df[column])
    return df2


def replace_with_one_hot_encoder(df: pd.DataFrame, column: str, ohe: OneHotEncoder,
                                 ohe_column_names: List[str]) -> pd.DataFrame:
    """
    This method should replace the column of df with all the columns generated from the one hot's version of the encoder
    Feel free to do it manually or through a sklearn ColumnTransformer
    :param df: Dataset
    :param column: column to be replaced
    :param ohe: the one hot encoder to be used to replace the column
    :param ohe_column_names: the names to be used as the one hot encoded's column names
    :return: The df with the column replaced with the one from label encoder
    """
    df2 = df.copy()

    ohe_values = ohe.fit_transform(df2[column].values.reshape(-1, 1)).toarray()

    # Traversing the column_names to insert columns into the df
    for i, col in enumerate(ohe_column_names):
        df2[col] = ohe_values[:, i]
    df2 = df2.drop(column, axis=1)

    return df2


def replace_label_encoder_with_original_column(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    """
    This method should revert what is done in replace_with_label_encoder
    The column of df should be from the label encoder, and you should use the le to revert the column to the previous state
    :param df: Dataset
    :param column: column to be replaced
    :param le: the label encoder to be used to revert the column
    :return: The df with the column reverted from label encoder
    """
    df2 = df.copy()
    df2[column] = le.inverse_transform(df2[column])

    return df2



def replace_one_hot_encoder_with_original_column(df: pd.DataFrame,
                                                 columns: List[str],
                                                 ohe: OneHotEncoder,
                                                 original_column_name: str) -> pd.DataFrame:
    """
    This method should revert what is done in replace_with_one_hot_encoder
    The columns (one of the method's arguments) are the columns of df that were placed there by the OneHotEncoder.
    You should use the ohe to revert these columns to the previous state (single column) which was present previously
    :param df: Dataset
    :param columns: the one hot encoded columns to be replaced
    :param ohe: the one hot encoder to be used to revert the columns
    :param original_column_name: the original column name which was used before being replaced with the one hot encoded version of it
    :return: The df with the columns reverted from the one hot encoder
    """

    df2 = df.copy()
    ohe_columns = df[columns]
    ohe_array = ohe.inverse_transform(ohe_columns)
    df2[original_column_name] = ohe_array
    df2= df2.drop(ohe_columns, 1)

    return df2


if __name__ == "__main__":
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [True, True, False, False], 'c': ['one', 'two', 'three', 'four']})
    le = generate_label_encoder(df.loc[:, 'c'])
    assert le is not None
    ohe = generate_one_hot_encoder(df.loc[:, 'c'])
    assert ohe is not None
    assert replace_with_label_encoder(df, 'c', le) is not None
    assert replace_with_one_hot_encoder(df, 'c', ohe, list(ohe.get_feature_names())) is not None
    assert replace_label_encoder_with_original_column(replace_with_label_encoder(df, 'c', le), 'c', le) is not None
    assert replace_one_hot_encoder_with_original_column(replace_with_one_hot_encoder(df, 'c', ohe, list(ohe.get_feature_names())),
                                                        list(ohe.get_feature_names()),
                                                        ohe,
                                                        'c') is not None
    print("ok")