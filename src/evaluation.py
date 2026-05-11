"""
Evaluation metrics for forecasting models.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    non_zero_mask = y_true != 0
    mape = np.mean(
        np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
    ) * 100

    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "MAPE": round(mape, 2)
    }

def highlight_table(s):
    sorted_vals = s.dropna().sort_values()
    min_val = sorted_vals.iloc[0]
    second_val = sorted_vals.iloc[1] if len(sorted_vals) > 1 else None

    styles = []
    for v in s:
        if v == min_val:
            styles.append("font-weight: bold")
        elif v == second_val:
            styles.append("text-decoration: underline")
        else:
            styles.append("")
    return styles