"""
Evaluation metrics for forecasting models.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
