import pandas as pd
from pathlib import Path

RAW_DATA_CSV = Path("data/raw/electricity_data.csv")


def load_raw_data(nrows: int | None = None) -> pd.DataFrame:
    base_path = Path(__file__).resolve().parents[1]
    full_path = base_path / RAW_DATA_CSV

    print(f"👉 Loading: {full_path}")

    if not full_path.exists():
        raise FileNotFoundError(f"❌ File not found: {full_path}")

    # ===== READ CSV =====
    df = pd.read_csv(
        full_path,
        sep=",",         
        header=0,
        nrows=nrows,
        low_memory=False
    )

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

    # ===== REPORT =====
    num_customers = df.shape[1]

    print("✅ Data loaded successfully!")
    print(f"📊 Samples: {df.shape[0]}")
    print(f"👥 Number of customers: {num_customers}")
    print(f"🕒 Time span: {df.index.min()} → {df.index.max()}")

    return df


