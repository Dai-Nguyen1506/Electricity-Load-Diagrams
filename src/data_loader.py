"""
Data loading utilities for electricity load dataset.
"""
import pandas as pd
from pathlib import Path
import os

# Đường dẫn tương đối từ thư mục gốc của dự án
RAW_DATA_PATH = Path("data/raw/electricity_data.csv")

def load_raw_data() -> pd.DataFrame:
    """
    Nạp dữ liệu thô, tự động xử lý Index thành kiểu Timestamp chuẩn.
    Kết quả: Index là 'timestamp' (datetime64), các cột bắt đầu từ 'MT_001'.
    """
    # 1. Xử lý đường dẫn
    try:
        base_path = Path(__file__).parent.parent
    except NameError:
        base_path = Path(os.getcwd())
        
    full_path = base_path / RAW_DATA_PATH

    if not full_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tại {full_path}")
        
    # 2. Nạp dữ liệu
    # index_col=0: Đưa cột thời gian vào Index ngay lập tức
    # parse_dates=True: Thử convert sang datetime ngay khi đọc
    df = pd.read_csv(
        full_path, 
        sep=';', 
        decimal=',', 
        index_col=0, 
        parse_dates=True, 
        low_memory=False
    )
    
    # 3. Ép kiểu dữ liệu Index (Phòng trường hợp parse_dates không tự nhận diện được format lạ)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 4. Đặt tên Index và sắp xếp
    df.index.name = 'timestamp'
    df.sort_index(inplace=True)
    
    return df