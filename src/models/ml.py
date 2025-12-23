"""
Machine learning models for forecasting.
"""

from sklearn.ensemble import RandomForestRegressor

def train_random_forest(X_train, y_train, **kwargs):
    model = RandomForestRegressor(**kwargs)
    model.fit(X_train, y_train)
    return model
