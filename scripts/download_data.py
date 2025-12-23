import os
import requests
import zipfile
import io
import pandas as pd

def download_and_convert():
    # 1. Cấu hình đường dẫn
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip"
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    txt_path = os.path.join(raw_dir, "LD2011_2014.txt")
    csv_path = os.path.join(raw_dir, "electricity_data.csv")

    # 2. Tải và giải nén (nếu chưa có file txt)
    if not os.path.exists(txt_path):
        print("⏳ Đang tải dữ liệu từ UCI (250MB ZIP)...")
        try:
            response = requests.get(url)
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(raw_dir)
            print("✅ Đã giải nén thành công file .txt")
        except Exception as e:
            print(f"❌ Lỗi tải file: {e}")
            return

    # 3. Chuyển đổi sang CSV chuẩn
    print("🔄 Đang chuyển đổi định dạng .txt sang .csv chuẩn...")
    print("⚠️ Lưu ý: File gốc khá lớn (1.2GB), quá trình này có thể mất 1-2 phút tùy vào RAM máy bạn.")
    
    try:
        # Đọc file với cấu hình của UCI: sep=';', decimal=','
        # low_memory=False giúp xử lý dữ liệu lớn ổn định hơn
        df = pd.read_csv(txt_path, sep=';', decimal=',', low_memory=False)
        
        # Lưu thành CSV chuẩn: sep=',', decimal='.'
        df.to_csv(csv_path, index=False)
        
        print(f"✅ HOÀN THÀNH! File CSV sẵn sàng tại: {csv_path}")
        print(f"📊 Thông tin dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột.")
        
        # Tùy chọn: Xóa file .txt cũ để tiết kiệm bộ nhớ
        # os.remove(txt_path)
        # print("🗑️ Đã xóa file .txt tạm thời để tiết kiệm ổ cứng.")

    except Exception as e:
        print(f"❌ Lỗi khi chuyển đổi: {e}")

if __name__ == "__main__":
    download_and_convert()