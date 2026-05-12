from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

import numpy as np


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = (
        mean_absolute_percentage_error(y_true, y_pred)
        * 100
    )

    results = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE (%)": mape
    }

    print("=" * 30)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print("=" * 30)

    return results