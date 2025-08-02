import os
import joblib
import json
import numpy as np
import sys
from sklearn.metrics import r2_score, mean_squared_error
from src.utils import load_data
from sklearn.linear_model import LinearRegression

try:
    # Load everything
    X, y = load_data()
    model = joblib.load("models/model.joblib")
    quant_model = joblib.load("models/model_quantized.joblib")

    with open("models/metrics.json") as f:
        metrics = json.load(f)
    with open("models/predictions.json") as f:
        preds = json.load(f)["predictions"]

    # Convert predictions to array
    preds = np.array(preds)

    # UAT checks (20 items)
    assert isinstance(model, LinearRegression)
    assert model.coef_ is not None
    assert isinstance(metrics["R2_score"], float)
    assert metrics["R2_score"] > 0.5
    assert metrics["MSE"] < 1.0
    assert os.path.exists("models/metrics_plot.png")
    assert preds.ndim == 1
    assert len(preds) == len(y)
    assert np.std(preds) > 0.1
    assert np.mean(preds) > 0.5
    assert np.max(preds) < 10
    assert np.min(preds) > 0
    assert not np.isnan(preds).any()
    assert not np.isinf(preds).any()
    assert isinstance(quant_model, LinearRegression)
    assert np.any(np.abs(model.coef_) > 0.01)
    assert isinstance(metrics["MSE"], float)
    assert X.shape[1] == len(model.coef_)
    assert len(set(preds.round())) > 5
    assert np.median(preds) > 1

    print("All 20 UAT tests passed.")

except AssertionError as e:
    print("UAT failed:", e)
    sys.exit(1)
except Exception as e:
    print("Unexpected error during UAT:", e)
    sys.exit(1)
