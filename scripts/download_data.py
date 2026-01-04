import os
import requests
import zipfile
import io
import pandas as pd
from requests.exceptions import RequestException, Timeout

def download_and_convert():
    # 1. Configure paths
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    txt_path = os.path.join(raw_dir, "LD2011_2014.txt")
    parquet_path = "data/raw/electricity_data.parquet"

    # 2. Download and extract (if txt file does not exist)
    if not os.path.exists(txt_path):
        print("⚠️  WARNING:")
        print("   The UCI server may block or throttle large downloads.")
        print("   If the program hangs or fails, please retry later or use a VPN.\n")

        print("⏳ Downloading data from UCI (250MB ZIP)...")

        try:
            response = requests.get(url, stream=True, timeout=(10, 60))
            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(raw_dir)

            print("✅ Successfully extracted the .txt file")

        except Timeout:
            print("❌ Download timed out.")
            print("   The UCI server may be slow or blocking your connection.")
            print("   Try again later or download the file manually.")

        except RequestException as e:
            print("❌ Download failed due to a network error.")
            print("   This often happens when the UCI server blocks access.")
            print(f"   Error details: {e}")

        except zipfile.BadZipFile:
            print("❌ Downloaded file is corrupted or incomplete.")
            print("   This usually means the response ended prematurely.")

        except Exception as e:
            print(f"❌ Unexpected error: {e}")

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