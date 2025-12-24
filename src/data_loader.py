import pandas as pd
from pathlib import Path
import os

# Đường dẫn tương đối từ thư mục gốc của dự án
RAW_DATA_PATH = Path("data/raw/electricity_data.csv")

def load_raw_data() -> pd.DataFrame:
    """
    Nạp dữ liệu thô và xử lý lỗi định dạng ngày tháng/dấu phân cách.
    """
    try:
        base_path = Path(__file__).parent.parent
    except NameError:
        base_path = Path(os.getcwd())
        
    full_path = base_path / RAW_DATA_PATH

    if not full_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file tại {full_path}")
        
    # 1. Đọc thử 1 dòng để kiểm tra separator nếu cần
    df = pd.read_csv(
        full_path,
        sep=",",
        index_col=0,                      
        low_memory=False
    )
    
    # 2. Xử lý Index sau khi nạp (An toàn hơn parse_dates trong read_csv)
    # Ép kiểu index về datetime, bỏ qua các lỗi nếu có để debug
    try:
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S", errors='coerce')
    except Exception:
        # Nếu format trên lỗi, để pandas tự suy luận
        df.index = pd.to_datetime(df.index, errors='coerce')

    # Loại bỏ các dòng không convert được ngày tháng (nếu có dòng header thừa)
    df = df[df.index.notnull()]
    
    # 3. Đặt tên Index và sắp xếp
    df.index.name = 'timestamp'
    df.sort_index(inplace=True)

    return df