# modules/predictor.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .contracts import IPredictor

class Predictor(IPredictor):
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y) -> None:
        self.model.fit(X, y)

    def validate(self, X_val, y_val) -> dict[str, float]:
        preds = self.model.predict(X_val)

        # Ensure 1-D arrays for metrics
        y_val = np.asarray(y_val).ravel()
        preds = np.asarray(preds).ravel()

        mae = mean_absolute_error(y_val, preds)

        # Backward-compatible RMSE (some sklearn versions don’t support squared=False)
        try:
            rmse = mean_squared_error(y_val, preds, squared=False)  # newer sklearn
        except TypeError:
            rmse = mean_squared_error(y_val, preds) ** 0.5          # older sklearn

        return {"MAE": float(mae), "RMSE": float(rmse)}

    def forecast(self, X_future) -> np.ndarray:
        return self.model.predict(X_future)
