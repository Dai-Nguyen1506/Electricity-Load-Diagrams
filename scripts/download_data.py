"""
Script to download Electricity Load Diagrams dataset
from Hugging Face and save as CSV.
"""

from datasets import load_dataset
import pandas as pd
from pathlib import Path


def download_data():
    print("Downloading dataset from Hugging Face...")

    dataset = load_dataset(
        "tulipa762/electricity_load_diagrams",
        split="train",
        trust_remote_code=True
    )

    print("Converting to pandas DataFrame...")
    df = dataset.to_pandas()

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "electricity.csv"
    df.to_csv(output_file, index=False)

    print(f"Dataset saved to {output_file}")


if __name__ == "__main__":
    download_data()
