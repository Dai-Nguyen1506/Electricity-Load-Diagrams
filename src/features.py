"""
Feature engineering for time series forecasting.
"""

def add_time_features(df):
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["month"] = df.index.month
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    return df

def add_lag_features(df, lags=(1, 24, 168)):
    for lag in lags:
        df[f"lag_{lag}"] = df["load"].shift(lag)
    return df

def add_rolling_features(df):
    df["rolling_24_mean"] = df["load"].rolling(24).mean()
    df["rolling_168_mean"] = df["load"].rolling(168).mean()
    return df
