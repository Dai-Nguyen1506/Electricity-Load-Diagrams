import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

TIME_SPLIT = "2014-01-01"

def seasonal_naive_model(data, target, seasonal_length):
    data = data.copy()
    if 'Timestamp' in data.columns:
        data = data.set_index('Timestamp')
    
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    val = data[data.index >= TIME_SPLIT].copy()

    pred_col = "prediction"

    # lấy đúng seasonal pattern trước đó
    val[pred_col] = data[target].iloc[
        -len(val)-seasonal_length:-seasonal_length
    ].values

    val = val.dropna(subset=[target, pred_col])

    mae = mean_absolute_error(val[target], val[pred_col])
    rmse = np.sqrt(mean_squared_error(val[target], val[pred_col]))
    mape = mean_absolute_percentage_error(val[target], val[pred_col]) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    return val, metrics

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer


def lightgbm_model(data, target, features):

    data = data.copy()

    if 'Timestamp' in data.columns:
        data = data.set_index('Timestamp')

    data.index = pd.to_datetime(data.index)

    data = data.dropna().sort_index()

    cat_cols = data[features].select_dtypes(
        include=["object", "category"]
    ).columns

    for col in cat_cols:
        data[col] = data[col].astype("category")

    train = data[data.index < TIME_SPLIT].copy()
    val = data[data.index >= TIME_SPLIT].copy()

    X_train, X_val = train[features], val[features]
    y_train, y_val = train[target], val[target]

    base_model = lgb.LGBMRegressor(
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )

    model = TransformedTargetRegressor(
        regressor=base_model,
        transformer=FunctionTransformer(
            np.log1p,
            inverse_func=np.expm1
        )
    )

    model.fit(X_train, y_train)

    pred_col = "prediction"

    val[pred_col] = model.predict(X_val)

    mae = mean_absolute_error(y_val, val[pred_col])

    rmse = np.sqrt(
        mean_squared_error(y_val, val[pred_col])
    )

    mape = mean_absolute_percentage_error(
        y_val,
        val[pred_col]
    ) * 100

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

    return val, model, metrics