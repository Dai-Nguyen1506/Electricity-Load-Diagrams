"""
Machine learning models for forecasting.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

def random_forest_model(X_train, y_train, X_test, params=None):

    default_params = {
        "random_state": 42,
        "n_estimators": 1000,
        "max_depth": 40,
        "max_samples": 0.8,
        "max_features": 0.8,
        "n_jobs": -1
    }

    if params is not None:
        default_params.update(params)

    model = TransformedTargetRegressor(
        regressor=RandomForestRegressor(**default_params),
        transformer=FunctionTransformer(
            np.log1p,
            inverse_func=np.expm1
        )
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return model, preds

def lightgbm_model(X_train, y_train, X_test, params=None):

    default_params = {
        "random_state": 42,
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "verbose": -1,
        "n_jobs": -1
    }

    if params is not None:
        default_params.update(params)

    model = TransformedTargetRegressor(
        regressor=LGBMRegressor(**default_params),
        transformer=FunctionTransformer(
            np.log1p,
            inverse_func=np.expm1
        )
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return model, preds

def xgboost_model(X_train, y_train, X_test, params=None):

    default_params = {
        "random_state": 42,
        "n_estimators": 1000,
        "learning_rate": 0.01,
        "max_depth": 40,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
        "n_jobs": -1
    }

    if params is not None:
        default_params.update(params)

    model = TransformedTargetRegressor(
        regressor=XGBRegressor(**default_params),
        transformer=FunctionTransformer(
            np.log1p,
            inverse_func=np.expm1
        )
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return model, preds