import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series):
    """Kiểm định tính dừng ADF Test"""
    result = adfuller(series.dropna())
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Is Stationary': result[1] < 0.05
    }

def handle_outliers(df, col, threshold=3):
    """Xử lý ngoại lệ bằng Z-score hoặc kẹp giá trị (Clipping)"""
    mean = df[col].mean()
    std = df[col].std()
    lower = mean - threshold * std
    upper = mean + threshold * std
    df[col] = df[col].clip(lower, upper)
    return df

def clean_and_prepare(df):
    # Nội suy các điểm mất điện/thiếu dữ liệu
    df = df.interpolate(method='time')
    return df