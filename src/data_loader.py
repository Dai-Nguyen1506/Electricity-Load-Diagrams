import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset

def load_from_gluonts():
    """Tải dữ liệu electricity từ GluonTS [cite: 10]"""
    dataset = get_dataset("electricity")
    return dataset.train, dataset.test

def load_local_txt(file_path):
    """Đọc file LD2011_2014.txt [cite: 11]"""
    # Cần xử lý định dạng dấu thập phân và dấu phẩy của file gốc
    return pd.read_csv(file_path, sep=';', decimal=',')