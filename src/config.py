import os

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models") # Tạo thêm thư mục này để lưu .pkl

# Tham số thời gian
TIME_COL = "timestamp"
TARGET_COL = "consumption" 
FREQ = "15T"              # Dữ liệu gốc 15 phút
S_PERIODS = [96, 672]     # 96 bước (1 ngày), 672 bước (1 tuần)

# Cấu hình dự báo
FORECAST_HORIZON = 96     # Dự báo trước 1 ngày (96 bước 15p)
SEED = 42