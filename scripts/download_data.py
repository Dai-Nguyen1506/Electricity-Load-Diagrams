import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import download_hf_dataset, save_raw_data

if __name__ == "__main__":
    df = download_hf_dataset()
    save_raw_data(df)
    print("Xong! Dữ liệu đã được lưu vào data/raw/")