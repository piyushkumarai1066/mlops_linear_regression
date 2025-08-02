import os
import json
import joblib
import numpy as np
import logging
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

logging.info("Loading original dataset...")
X, y = fetch_california_housing(return_X_y=True)
logging.info(f"Total records loaded: {X.shape[0]} samples with {X.shape[1]} features.")

logging.info("Quantizing input to float16...")
X_quantized = X.astype(np.float16)

logging.info("Loading trained model...")
model = joblib.load("models/model.joblib")

logging.info("Generating predictions on full quantized dataset...")
predictions = model.predict(X_quantized)

r2 = r2_score(y, predictions)
mse = mean_squared_error(y, predictions)

logging.info(f"R² Score (quantized): {r2:.4f}")
logging.info(f"MSE (quantized): {mse:.4f}")
logging.info(f"Sample predictions: {np.round(predictions[:5], 3)}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model_quantized.joblib")
logging.info("Quantized model saved to models/model_quantized.joblib")

quant_metrics = {
    "R2_score": round(r2, 4),
    "MSE": round(mse, 4),
    "sample_predictions": predictions[:5].tolist()
}

with open("models/quant_metrics.json", "w") as f:
    json.dump(quant_metrics, f, indent=2)

logging.info("Quantized evaluation metrics saved to models/quant_metrics.json")
