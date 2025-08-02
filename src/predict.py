import joblib
import logging
import json
from sklearn.datasets import fetch_california_housing

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def run_prediction():
    logging.info("Loading model from models/model.joblib...")
    model = joblib.load("models/model.joblib")
    logging.info("Fetching California housing data...")
    X, _ = fetch_california_housing(return_X_y=True)
    logging.info("Generating predictions on first 5 samples...")
    preds = model.predict(X[:5])
    logging.info(f"Sample predictions: {preds}")
    output = {"predictions": preds.tolist()}
    with open("models/predictions.json", "w") as f:
        json.dump(output, f, indent=2)
    logging.info("Predictions saved to models/predictions.json")

if __name__ == "__main__":
    run_prediction()
