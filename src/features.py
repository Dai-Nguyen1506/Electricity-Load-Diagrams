import numpy as np
import pandas as pd
import holidays

def add_fourier_features(df, periods=[96, 672], n_order=5):
    """Tạo các biến Sin/Cos để bắt chu kỳ ngày và tuần"""
    for period in periods:
        for i in range(1, n_order + 1):
            df[f'fourier_sin_{period}_{i}'] = np.sin(2 * np.pi * i * df.index.hour / period)
            df[f'fourier_cos_{period}_{i}'] = np.cos(2 * np.pi * i * df.index.hour / period)
    return df

def add_holiday_features(df, country='VN'):
    """Đánh dấu các ngày lễ (ảnh hưởng mạnh đến tải điện)"""
    vn_holidays = holidays.CountryHoliday(country)
    df['is_holiday'] = df.index.map(lambda x: 1 if x in vn_holidays else 0)
    return df

def create_advanced_features(df):
    df = add_fourier_features(df)
    df = add_holiday_features(df)
    # Lag features cho mô hình ML
    for lag in [1, 96, 672]:
        df[f'lag_{lag}'] = df['consumption'].shift(lag)
    return df.dropna()