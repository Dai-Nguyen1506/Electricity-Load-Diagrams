"""
Statistical forecasting models.
"""

from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarima(train, exog=None, order=None, seasonal_order=None):
    model = SARIMAX(
        train,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit()
