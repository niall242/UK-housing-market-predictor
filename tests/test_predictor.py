# tests/test_predictor.py
import numpy as np
from modules.predictor import Predictor


def make_tiny_series(n=10):
    """
    Create a simple linear trend: y = 2x + 5
    Returns X, y, X_val, y_val
    """
    x = np.arange(n).reshape(-1, 1)
    y = 2 * x.ravel() + 5
    split = n - 3
    return x[:split], y[:split], x[split:], y[split:]


def test_train_and_validate_returns_metrics():
    X_train, y_train, X_val, y_val = make_tiny_series()
    predictor = Predictor()

    predictor.train(X_train, y_train)
    metrics = predictor.validate(X_val, y_val)

    # Both keys should exist and be floats
    assert "MAE" in metrics and "RMSE" in metrics
    assert all(isinstance(v, float) for v in metrics.values())

    # Should be near-zero for this perfect linear case
    assert metrics["MAE"] < 1e-9
    assert metrics["RMSE"] < 1e-9


def test_forecast_returns_expected_shape():
    X_train, y_train, _, _ = make_tiny_series()
    predictor = Predictor()
    predictor.train(X_train, y_train)

    X_future = np.arange(10, 15).reshape(-1, 1)
    preds = predictor.forecast(X_future)

    # 5 predictions, shape (5,)
    assert preds.shape == (5,)
    # Roughly follow y = 2x + 5
    assert np.allclose(preds, 2 * X_future.ravel() + 5, atol=1e-9)
