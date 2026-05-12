"""
Baseline forecasting models.
"""
import pandas as pd

def naive_forecast(train, test):
    last_value = train.iloc[-1, 0]
    return pd.Series([last_value] * len(test), index=test.index)

def seasonal_naive_forecast(df, test, season=7):
    preds = [df.loc[t - pd.Timedelta(hours=24 * season), 'total_load'] for t in test.index]
    return pd.Series(preds, index=test.index)
