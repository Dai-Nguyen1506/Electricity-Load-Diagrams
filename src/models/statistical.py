"""
Statistical forecasting models.
"""

from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarima(series, order, seasonal_order):
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit()
