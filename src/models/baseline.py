"""
Baseline forecasting models.
"""

def naive_forecast(series):
    return series.shift(1)

def seasonal_naive_forecast(series, seasonality=24):
    return series.shift(seasonality)
