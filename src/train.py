import os
import joblib
import logging
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.utils import load_data

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def train_and_save_model():
    logging.info("Loading dataset...")
    X, y = load_data()

    logging.info("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X, y)

    logging.info("Evaluating model performance...")
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    mse = mean_squared_error(y, preds)

    logging.info(f"R² Score: {r2}")
    logging.info(f"MSE: {mse}")

    metrics = {
        "R2_score": round(r2, 4),
        "MSE": round(mse, 4)
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")
    logging.info("Model saved to models/model.joblib")

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Metrics saved to models/metrics.json")

if __name__ == "__main__":
    train_and_save_model()
