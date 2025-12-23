from lightgbm import LGBMRegressor
import joblib
from src.config import MODEL_DIR

class GlobalForecaster:
    def __init__(self):
        self.model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31
        )
        
    def train(self, X, y):
        self.model.fit(X, y)
        
    def save(self, name="lgbm_v1.pkl"):
        joblib.dump(self.model, f"{MODEL_DIR}/{name}")

    def load(self, name="lgbm_v1.pkl"):
        self.model = joblib.load(f"{MODEL_DIR}/{name}")