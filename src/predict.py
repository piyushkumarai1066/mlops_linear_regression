import joblib
import logging
import json
from sklearn.datasets import fetch_california_housing
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def run_prediction():
    logging.info("Loading model from models/model.joblib...")
    model = joblib.load("models/model.joblib")

    logging.info("Fetching California housing data...")
    X, _ = fetch_california_housing(return_X_y=True)

    logging.info(f"Generating predictions for all {X.shape[0]} samples...")
    preds = model.predict(X)

    summary = {
        "total_samples": len(preds),
        "min_prediction": round(preds.min(), 2),
        "max_prediction": round(preds.max(), 2),
        "mean_prediction": round(preds.mean(), 2)
    }

    logging.info(f"Prediction summary: {summary}")
    logging.info(f"First 5 predictions: {np.round(preds[:5], 3)}")

    output = {
        "predictions": preds.tolist(),
        "summary": summary
    }

    with open("models/predictions.json", "w") as f:
        json.dump(output, f, indent=2)

    logging.info("Full predictions saved to models/predictions.json")

if __name__ == "__main__":
    run_prediction()
