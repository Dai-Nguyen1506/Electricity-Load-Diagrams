"""
Time-series splitting utilities.
"""
import pandas as pd

def time_train_test_split(df, split_date="2014-01-01"):
    df = df.sort_index()
    
    train = df.loc[df.index < split_date]
    test = df.loc[df.index >= split_date]
    
    return train, test
