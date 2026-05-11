"""
Feature engineering for time series forecasting.
"""

def add_time_features(df):
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["month"] = df.index.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    return df

def add_lag_features(df, lags=(1, 24, 168), target_col="load"):
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df

def add_rolling_features(df, sizes=(24, 168), target_col="load"):
    for size in sizes:
        df[f"rolling_{size}_mean"] = df[target_col].rolling(size).mean()
        df[f"rolling_{size}_std"] = df[target_col].rolling(size).std()
    return df
