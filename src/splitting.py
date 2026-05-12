"""
Time-series splitting utilities.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

def time_train_test_split(df, split_date="2014-01-01"):
    df = df.sort_index()
    
    train = df.loc[df.index < split_date]
    test = df.loc[df.index >= split_date]
    
    return train, test

def scaler (train, test):
    scaler = StandardScaler()
    scaler.fit(train)
    
    train_scaled = pd.DataFrame(scaler.transform(train), index=train.index, columns=train.columns)
    test_scaled = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)
    
    return train_scaled, test_scaled

def split_train_test(df, target, split_time, categorical_cols=None):

    train_df = df.loc[:split_time].copy()
    test_df = df.loc[split_time:].iloc[1:].copy()

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.drop(columns=target)
    y_test = test_df[target]

    encoder = None

    if categorical_cols:
        if isinstance(categorical_cols, str):
            cols_to_encode = [categorical_cols]
        else:
            cols_to_encode = categorical_cols
            
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[cols_to_encode] = encoder.fit_transform(X_train[cols_to_encode])
        X_test[cols_to_encode] = encoder.transform(X_test[cols_to_encode])

    return X_train, X_test, y_train, y_test, encoder