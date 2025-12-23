import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_mase(y_true, y_pred, y_train, sp=96):
    """
    MASE = MAE / MAE_naive_seasonal
    sp: seasonal period (96 cho chu kỳ ngày)
    """
    mae = mean_absolute_error(y_true, y_pred)
    # Tính sai số của mô hình Naive (lấy giá trị chu kỳ trước làm dự báo)
    naive_forecast_error = np.mean(np.abs(np.diff(y_train, n=sp)))
    return mae / naive_forecast_error

def evaluate_all(y_true, y_pred, y_train):
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MASE": calculate_mase(y_true, y_pred, y_train)
    }
    return metrics