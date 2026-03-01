import os
import pandas as pd
from pathlib import Path

def load_data(file_path: str, nrows: int | None = None) -> pd.DataFrame:
    base_path = Path(__file__).resolve().parents[1]
    full_path = base_path / file_path

    if not full_path.exists():
        raise FileNotFoundError(f"❌ File not found: {full_path}")
    
    print(f"👉 Loading: {file_path}")
    
    # ===== READ PARQUET =====
    df = pd.read_parquet(full_path)

    if nrows is not None:
        df = df.head(nrows)

    return df


RAW_DATA_PARQUET = Path("data/raw/electricity_data.parquet")

def load_raw_data(nrows: int | None = None) -> pd.DataFrame:
    df = load_data(RAW_DATA_PARQUET, nrows)

    #  timestamp
    timestamp_col = df.columns[0]

    df.rename(columns={timestamp_col: "Timestamp"}, inplace=True)

    # ===== PARSE TIME =====
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    df = df.dropna(subset=["Timestamp"])
    df = df.set_index("Timestamp").sort_index()

    # ===== CAST DATA =====
    df = df.astype("float64")

    return df

def save_file(df, file_path):
    save_dir = "../data/processed"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_path)
    df.to_parquet(save_path, index=True)
    print(f"File saved as: {save_path}")