"""
Deep learning models for forecasting.
"""
import numpy as np
import holidays
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense, Dropout

def dl_forecast(train, test, window=30):
    # scale data
    scaler = RobustScaler()
    train_scaled = scaler.fit_transform(train)

    # build sequences
    HORIZON = len(test)

    def create_seq(data):
        X, y = [], []
        for i in range(len(data) - window - HORIZON):
            X.append(data[i:i + window])
            y.append(data[i + window:i + window + HORIZON, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_seq(train_scaled)

    # build model
    model = Sequential([
        Input(shape=(window, train_scaled.shape[1])),

        GRU(128, return_sequences=True),
        Dropout(0.2),

        GRU(64),
        Dropout(0.2),

        Dense(128, activation="relu"),
        Dense(HORIZON)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="huber"
    )

    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        verbose=0
    )

    # forecast
    last_window = train_scaled[-window:]

    future_scaled = model.predict(
        last_window.reshape(1, window, train_scaled.shape[1]),
        verbose=0
    )[0]

    # inverse scaling safely
    dummy = np.zeros((HORIZON, train_scaled.shape[1]))
    dummy[:, 0] = future_scaled

    pred = scaler.inverse_transform(dummy)[:, 0]


    return pred

def dl_features(df):
    df_features = df.copy()

    df_features["month"] = df_features.index.month
    df_features["day_of_week"] = df_features.index.dayofweek
    df_features["day_of_year"] = df_features.index.dayofyear

    df_features["is_weekend"] = (df_features["day_of_week"] >= 5).astype(int)

    df_features["day_sin"] = np.sin(2 * np.pi * df_features["day_of_year"] / 365)
    df_features["day_cos"] = np.cos(2 * np.pi * df_features["day_of_year"] / 365)

    df_features["month_sin"] = np.sin(2 * np.pi * df_features["month"] / 12)
    df_features["month_cos"] = np.cos(2 * np.pi * df_features["month"] / 12)

    pt_holidays = holidays.Portugal(years=range(2011, 2014))

    df_features["holiday_flag"] = (
        df_features.index.normalize()
        .map(lambda d: 1 if d in pt_holidays else 0)
    )

    return df_features