def handle_missing_values(df):
    """Xử lý giá trị thiếu [cite: 29]"""
    return df.fillna(method='ffill')

def create_time_features(df):
    """Tạo các tính năng thời gian: giờ, ngày, tháng, mùa [cite: 31, 35, 36]"""
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    return df