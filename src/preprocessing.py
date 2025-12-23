"""
Data preprocessing functions.
"""

import pandas as pd

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values using forward fill.
    """
    return df.fillna(method="ffill")

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle outliers (optional).
    """
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    """
    df = handle_missing_values(df)
    df = handle_outliers(df)
    return df
