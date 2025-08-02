import os
import sys
import json
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

def fail(msg):
    print(f" UAT failed: {msg}")
    sys.exit(1)

try:
    print("Loading dataset and model artifacts...")
    X, y = fetch_california_housing(return_X_y=True)

    # Existence checks
    required_files = [
        "models/model.joblib",
        "models/model_quantized.joblib",
        "models/predictions.json",
        "models/metrics.json",
        "models/quant_metrics.json",
        "models/metrics_plot.png"
    ]
    for file in required_files:
        if not os.path.exists(file):
            fail(f"{file} not found.")

    model = joblib.load("models/model.joblib")
    quant_model = joblib.load("models/model_quantized.joblib")

    with open("models/predictions.json") as f:
        preds = np.array(json.load(f)["predictions"])

    with open("models/metrics.json") as f:
        metrics = json.load(f)

    with open("models/quant_metrics.json") as f:
        qmetrics = json.load(f)

    print("✅ Artifacts loaded successfully.")

    # UAT Checks (20)
    assert isinstance(model, LinearRegression), "Model is not LinearRegression"
    assert isinstance(quant_model, LinearRegression), "Quantized model is not LinearRegression"
    assert isinstance(metrics["R2_score"], float), "R2_score missing or not float"
    assert isinstance(metrics["MSE"], float), "MSE missing or not float"
    assert isinstance(qmetrics["R2_score"], float), "Quantized R2 missing or not float"
    assert isinstance(qmetrics["MSE"], float), "Quantized MSE missing or not float"
    assert len(model.coef_) == X.shape[1], "Model coefficients mismatch feature shape"
    assert len(preds) == len(y), f"Prediction count mismatch: {len(preds)} vs {len(y)}"
    assert preds.ndim == 1, "Predictions are not 1D"
    assert not np.isnan(preds).any(), "NaNs found in predictions"
    assert not np.isinf(preds).any(), "Infs found in predictions"
    assert preds.min() > 0, "Negative or zero predictions detected"
    assert preds.max() < 10, "Some predictions unusually large"
    assert np.std(preds) > 0.1, "Predictions are too flat"
    assert np.median(preds) > 1.0, "Median prediction too low"
    assert len(set(preds.round(1))) > 5, "Not enough prediction variability"
    assert metrics["R2_score"] > 0.5, f"R² too low: {metrics['R2_score']}"
    assert metrics["MSE"] < 1.0, f"MSE too high: {metrics['MSE']}"
    assert os.path.exists("models/metrics_plot.png"), "metrics_plot.png missing"
    assert os.path.exists("models/model_quantized.joblib"), "Quantized model missing"

    print("All 20 UAT checks passed successfully.")

except AssertionError as e:
    fail(str(e))
except Exception as e:
    fail(f"Unexpected error: {e}")
