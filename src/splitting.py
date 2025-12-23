"""
Time-series splitting utilities.
"""

def time_train_test_split(df, test_size=0.2):
    split_index = int(len(df) * (1 - test_size))
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test
