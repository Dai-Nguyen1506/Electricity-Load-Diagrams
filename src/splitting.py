"""
Time-series splitting utilities.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

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