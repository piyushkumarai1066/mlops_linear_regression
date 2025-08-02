import os
import sys
import json
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from src.utils import load_data
from sklearn.linear_model import LinearRegression

def fail(msg):
    print(f"UAT failed: {msg}")
    sys.exit(1)

try:
    print("🔍 Loading dataset and model artifacts...")
    X, y = load_data()

    if not os.path.exists("models/model.joblib"):
        fail("model.joblib not found.")

    if not os.path.exists("models/model_quantized.joblib"):
        fail("model_quantized.joblib not found.")

    if not os.path.exists("models/predictions.json"):
        fail("predictions.json not found.")

    if not os.path.exists("models/metrics.json"):
        fail("metrics.json not found.")

    if not os.path.exists("models/quant_metrics.json"):
        fail("quant_metrics.json not found.")

    model = joblib.load("models/model.joblib")
    quant_model = joblib.load("models/model_quantized.joblib")

    with open("models/predictions.json") as f:
        preds = np.array(json.load(f)["predictions"])

    with open("models/metrics.json") as f:
        metrics = json.load(f)

    with open("models/quant_metrics.json") as f:
        qmetrics = json.load(f)

    print("Artifacts loaded successfully.")

    # UAT checks
    assert isinstance(model, LinearRegression), "Loaded model is not LinearRegression."
    assert isinstance(metrics["R2_score"], float), "R2_score in metrics.json is not a float."
    assert isinstance(metrics["MSE"], float), "MSE in metrics.json is not a float."
    assert metrics["R2_score"] > 0.5, f"R² too low: {metrics['R2_score']}"
    assert metrics["MSE"] < 1.0, f"MSE too high: {metrics['MSE']}"
    assert len(preds) == len(y), "Prediction count mismatch."
    assert preds.ndim == 1, "Predictions not 1D."
    assert not np.isnan(preds).any(), "NaNs found in predictions."
    assert not np.isinf(preds).any(), "Infs found in predictions."
    assert preds.min() > 0, "Some predictions are negative or zero."
    assert preds.max() < 10, "Some predictions unusually large."
    assert np.std(preds) > 0.1, "Predictions are too flat."
    assert isinstance(qmetrics["R2_score"], float), "Quantized R² not a float."
    assert isinstance(qmetrics["MSE"], float), "Quantized MSE not a float."
    assert isinstance(quant_model, LinearRegression), "Quantized model is not LinearRegression."
    assert len(model.coef_) == X.shape[1], "Model coefficients mismatch input shape."
    assert len(set(preds.round(1))) > 5, "Not enough prediction variability."
    assert np.median(preds) > 1.0, "Median prediction is too low."
    assert os.path.exists("models/metrics_plot.png"), "metrics_plot.png missing."
    assert os.path.exists("models/model_quantized.joblib"), "Quantized model missing."

    print(" All 20 UAT checks passed successfully.")

except AssertionError as e:
    fail(str(e))
except Exception as e:
    fail(f"Unexpected error: {e}")
