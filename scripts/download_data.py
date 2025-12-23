"""
Script to download electricity load data manually or via HF.
"""

from datasets import load_dataset
import pandas as pd
from pathlib import Path

def download_data():
    dataset = load_dataset("tulipa762/electricity_load_diagrams", split="train")
    df = dataset.to_pandas()

    output_path = Path("data/raw")
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "electricity.csv", index=False)

if __name__ == "__main__":
    download_data()
