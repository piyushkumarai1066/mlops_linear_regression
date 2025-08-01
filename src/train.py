from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np  # Add this import
from src.utils import load_data

def train_and_save_model():
    X, y = load_data()
    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    r2 = r2_score(y, preds)
    mse = mean_squared_error(y, preds)

    print(f"R2 Score: {r2}")
    print(f"MSE: {mse}")
    
    # ✅ Add this to print sample predictions
    print("Sample predictions:", np.round(preds[:5], 3))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.joblib")

if __name__ == "__main__":
    train_and_save_model()
