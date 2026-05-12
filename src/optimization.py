import optuna
import numpy as np

from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


MODEL_CONFIGS = {

    "lightgbm": {
        "model_class": LGBMRegressor,
        "base_params": {
            "random_state": 42,
            "verbose": -1,
            "n_jobs": -1
        },
        "hyper_params": {
            "n_estimators": ("int", 300, 3000),
            "learning_rate": ("float", 0.005, 0.05, True),
            "max_depth": ("int", 3, 12),
            "num_leaves": ("int", 15, 255),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "min_child_samples": ("int", 5, 100),
            "reg_alpha": ("float", 0.0, 10.0),
            "reg_lambda": ("float", 0.0, 10.0)
        }
    },

    "xgboost": {
        "model_class": XGBRegressor,
        "base_params": {
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0
        },
        "hyper_params": {
            "n_estimators": ("int", 300, 3000),
            "learning_rate": ("float", 0.005, 0.05, True),
            "max_depth": ("int", 3, 12),
            "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.6, 1.0),
            "min_child_weight": ("int", 1, 20),
            "gamma": ("float", 0.0, 10.0),
            "reg_alpha": ("float", 0.0, 10.0),
            "reg_lambda": ("float", 0.0, 10.0)
        }
    },

    "catboost": {
        "model_class": CatBoostRegressor,
        "base_params": {
            "random_state": 42,
            "verbose": 0,
            "loss_function": "RMSE"
        },
        "hyper_params": {
            "iterations": ("int", 300, 3000),
            "learning_rate": ("float", 0.005, 0.05, True),
            "depth": ("int", 3, 12),
            "subsample": ("float", 0.6, 1.0),
            "l2_leaf_reg": ("float", 1.0, 10.0)
        }
    }
}


def _suggest_params(trial, hyper_params):

    params = {}

    for name, cfg in hyper_params.items():

        if cfg[0] == "int":
            params[name] = trial.suggest_int(name, cfg[1], cfg[2])

        elif cfg[0] == "float":
            params[name] = trial.suggest_float(
                name,
                cfg[1],
                cfg[2],
                log=cfg[3] if len(cfg) > 3 else False
            )

    return params


def optuna_optimize(
    X_train,
    y_train,
    X_valid,
    y_valid,
    model_name="lightgbm",
    n_trials=50
):

    config = MODEL_CONFIGS[model_name]

    model_class = config["model_class"]
    base_params = config["base_params"]
    hyper_params = config["hyper_params"]

    def objective(trial):

        params = {
            **base_params,
            **_suggest_params(trial, hyper_params)
        }

        model = model_class(**params)

        model.fit(X_train, y_train)

        preds = model.predict(X_valid)

        rmse = np.sqrt(
            mean_squared_error(y_valid, preds)
        )

        return rmse

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=n_trials)

    print("=" * 40)
    print(f"MODEL      : {model_name}")
    print(f"BEST RMSE  : {study.best_value:.4f}")
    print("BEST PARAMS:")
    print(study.best_params)
    print("=" * 40)

    return study


def optuna_lightgbm(
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_trials=50
):
    return optuna_optimize(
        X_train,
        y_train,
        X_valid,
        y_valid,
        model_name="lightgbm",
        n_trials=n_trials
    )


def optuna_xgboost(
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_trials=50
):
    return optuna_optimize(
        X_train,
        y_train,
        X_valid,
        y_valid,
        model_name="xgboost",
        n_trials=n_trials
    )


def optuna_catboost(
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_trials=50
):
    return optuna_optimize(
        X_train,
        y_train,
        X_valid,
        y_valid,
        model_name="catboost",
        n_trials=n_trials
    )