import os
import requests
import zipfile
import io
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def download_and_convert():
    # 1. Configure paths
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    txt_path = os.path.join(raw_dir, "LD2011_2014.txt")
    parquet_path = "data/raw/electricity_data.parquet"

    # 2. Download and extract (if txt file does not exist)
    if not os.path.exists(txt_path):
        print("⏳ Downloading data from UCI (250MB ZIP)...")
        try:
            response = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(raw_dir)
            print("✅ Successfully extracted the .txt file")
        except Exception as e:
            print(f"❌ Download error: {e}")
            return

    # 3. Convert to standard CSV
    print("🔄 Converting .txt format to standard .parquet...")
    
    try:
        # Read file using UCI format: sep=';', decimal=','
        # low_memory=False ensures more stable handling of large files
        df = pd.read_csv(txt_path, sep=';', decimal=',', low_memory=False)

        # Save as standard parquet
        df.to_parquet(parquet_path,engine="pyarrow",compression="snappy")
        
        print(f"✅ DONE! Parquet file is ready at: {parquet_path}")
        print(f"📊 Dataset info: {df.shape[0]} rows, {df.shape[1]} columns.")
        
        # Remove old .txt file to save disk space
        os.remove(txt_path)
        print("🗑️ Temporary .txt file deleted to save disk space.")

    except Exception as e:
        print(f"❌ Conversion error: {e}")

if __name__ == "__main__":
    download_and_convert()