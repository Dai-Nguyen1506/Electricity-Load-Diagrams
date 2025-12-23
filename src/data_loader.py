from datasets import load_dataset
import pandas as pd
from src.config import DATA_RAW

def download_hf_dataset(repo_id="tulipa762/electricity_load_diagrams"):
    print(f"Downloading {repo_id}...")
    dataset = load_dataset(repo_id)
    df = pd.DataFrame(dataset['train']) # Hoặc 'test' tùy cấu trúc HF
    return df

def save_raw_data(df, filename="electricity_raw.csv"):
    df.to_csv(f"{DATA_RAW}/{filename}", index=False)