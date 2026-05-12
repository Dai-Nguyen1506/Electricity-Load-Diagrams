import numpy as np

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# =====================================================
# LIGHTGBM
# =====================================================

def lightgbm_model(X_train, y_train, X_test, params=None):

    default_params = {
        "random_state": 42,
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
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


# =====================================================
# XGBOOST
# =====================================================

def xgboost_model(X_train, y_train, X_test, params=None):

    default_params = {
        "random_state": 42,
        "n_estimators": 1000,
        "learning_rate": 0.03,
        "max_depth": 40,
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


# =====================================================
# CATBOOST
# =====================================================

def catboost_model(X_train, y_train, X_test, params=None):

    default_params = {
        "random_state": 42,
        "iterations": 1000,
        "learning_rate": 0.03,
        "depth": 16,
        "verbose": 0,
        "thread_count": -1
    }

    if params is not None:
        default_params.update(params)

    model = TransformedTargetRegressor(
        regressor=CatBoostRegressor(**default_params),
        transformer=FunctionTransformer(
            np.log1p,
            inverse_func=np.expm1
        )
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return model, preds