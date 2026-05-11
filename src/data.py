import pandas as pd
from pathlib import Path

def get_data(filename):
    """
    Đọc dữ liệu đã được xử lý từ thư mục data/processed.
    filename: tên file (ví dụ: 'sales_features.parquet')
    Trả về DataFrame chứa dữ liệu đã đọc.
    """
    repo_root = Path(__file__).resolve().parents[1]
    processed_dir = repo_root / 'data' / 'processed'
    file_path = processed_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file {file_path}. Vui lòng kiểm tra lại tên file và thư mục.")

    try:
        df = pd.read_parquet(file_path)
        print(f"Đã đọc thành công dữ liệu từ: {file_path}")
        return df
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None
    
def save_data(data, filename):
	"""
	Lưu dữ liệu vào thư mục data/processed với tên file chỉ định.
	data: dữ liệu (DataFrame hoặc string)
	filename: tên file (ví dụ: 'output.parquet')
	"""
	repo_root = Path(__file__).resolve().parents[1]
	processed_dir = repo_root / 'data' / 'processed'
	processed_dir.mkdir(parents=True, exist_ok=True)
	file_path = processed_dir / filename

	# Nếu là pandas DataFrame thì dùng to_parquet, nếu là string thì ghi trực tiếp
	try:
		if hasattr(data, 'to_parquet'):
			data.to_parquet(file_path)
		else:
			file_path.write_text(str(data), encoding='utf-8')
		print(f"Đã lưu thành công tại: {file_path}")
	except Exception as e:
		print(f"Lỗi khi lưu file: {e}")

def get_raw():
	"""
	Đọc dữ liệu thô từ thư mục data/raw.
	Trả về DataFrame chứa dữ liệu đã đọc.
	"""
	repo_root = Path(__file__).resolve().parents[1]
	raw_dir = repo_root / 'data' / 'raw'
	file_path = raw_dir / 'electricity_data.parquet'

	if not file_path.exists():
		raise FileNotFoundError(f"Không tìm thấy file {file_path}. Vui lòng kiểm tra lại tên file và thư mục.")

	try:
		df = pd.read_parquet(file_path)

		df['Timestamp'] = pd.to_datetime(df['Unnamed: 0'])
		df = df.set_index('Timestamp')
		df = df.drop(columns=['Unnamed: 0'])
		df = df.sort_index()
		
		print(f"Đã đọc thành công dữ liệu thô từ: {file_path}")
		return df
	except Exception as e:
		print(f"Lỗi khi đọc file: {e}")
		return None