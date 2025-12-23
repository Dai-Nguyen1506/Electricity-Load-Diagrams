"""
Data loading utilities for electricity load dataset.
"""

import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/electricity.csv")

def load_raw_data() -> pd.DataFrame:
    """
    Load raw electricity load data.

    Returns
    -------
    pd.DataFrame
        Raw dataset
    """
    return pd.read_csv(RAW_DATA_PATH)
