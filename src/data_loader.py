import pandas as pd
from pathlib import Path
import os

# Relative path from the project root directory
RAW_DATA_PATH = Path("data/raw/electricity_data.csv")

def load_raw_data() -> pd.DataFrame:
    """
    Load raw data and handle date/time format and delimiter issues.
    """
    try:
        base_path = Path(__file__).parent.parent
    except NameError:
        base_path = Path(os.getcwd())
        
    full_path = base_path / RAW_DATA_PATH

    if not full_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tại {full_path}")
        
    # 1. Read the file (separator is fixed as ',' for converted CSV)
    df = pd.read_csv(
        full_path,
        sep=",",
        index_col=0,                      
        low_memory=False
    )
    
    # 2. Process index after loading (safer than using parse_dates in read_csv)
    # Force index to datetime, ignore errors for debugging purposes
    try:
        df.index = pd.to_datetime(
            df.index,
            format="%Y-%m-%d %H:%M:%S",
            errors='coerce'
        )
    except Exception:
        # If the format above fails, let pandas infer automatically
        df.index = pd.to_datetime(df.index, errors='coerce')

    # Drop rows where datetime conversion failed (e.g., extra header rows)
    df = df[df.index.notnull()]

    # 3. Set index name and sort by time 
    df.index.name = 'timestamp'
    df.sort_index(inplace=True)

    return df