import os
import json
import joblib
import numpy as np
import logging
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

logging.info("📦 Loading California housing dataset...")
X, y = fetch_california_housing(return_X_y=True)
logging.info(f"✅ Dataset shape: {X.shape}, Samples: {X.shape[0]}")

logging.info("🔧 Quantizing input to float16...")
X_quantized = X.astype(np.float16)

logging.info("📥 Loading trained model...")
model = joblib.load("models/model.joblib")

logging.info("⚙️  Running prediction on quantized data...")
predictions = model.predict(X_quantized)

r2 = r2_score(y, predictions)
mse = mean_squared_error(y, predictions)

logging.info(f"📊 R² Score (quantized): {r2:.4f}")
logging.info(f"📊 MSE (quantized): {mse:.4f}")
logging.info(f"🧾 Sample predictions: {np.round(predictions[:5], 3)}")

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model_quantized.joblib")
logging.info("💾 Quantized model saved as models/model_quantized.joblib")

quant_metrics = {
    "R2_score": round(r2, 4),
    "MSE": round(mse, 4),
    "sample_predictions": predictions[:5].tolist()
}

with open("models/quant_metrics.json", "w") as f:
    json.dump(quant_metrics, f, indent=2)

logging.info("📝 Quantized metrics saved to models/quant_metrics.json")

# 🔐 Fail-fast check
if not os.path.exists("models/quant_metrics.json"):
    raise FileNotFoundError("❌ quant_metrics.json was not created!")

logging.info("✅ quant_metrics.json successfully created.")
