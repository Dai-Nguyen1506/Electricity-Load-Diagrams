import pandas as pd
from pathlib import Path
import os

# Relative path from the project root directory
RAW_DATA_PATH = Path("data/raw/electricity_data.parquet")

def load_raw_data(columns: list[str] | None = None) -> pd.DataFrame:
    try:
        base_path = Path(__file__).parent.parent
    except NameError:
        base_path = Path(os.getcwd())
        
    full_path = base_path / RAW_DATA_PATH

    if not full_path.exists():
        raise FileNotFoundError(f"File not found at {full_path}")
        
    # Read Parquet
    df = pd.read_parquet(full_path, columns=columns)
    
    # Set Timestamp (first column) as index
    first_col = df.columns[0]
    df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
    df.set_index(first_col, inplace=True)

    # Drop invalid timestamps
    df = df[df.index.notnull()]

    # Set index name and sort
    df.index.name = "Timestamp"
    df.sort_index(inplace=True)

    return df