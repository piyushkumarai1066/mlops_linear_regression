from sklearn.metrics import mean_squared_error, r2_score
from src.utils import load_data
import joblib
import sys

X, y = load_data()
model = joblib.load("models/model.joblib")
preds = model.predict(X)

r2 = r2_score(y, preds)
mse = mean_squared_error(y, preds)

print(f"R² Score: {r2}")
print(f"MSE: {mse}")

if r2 < 0.5:
    print("R² score too low. Failing pipeline.")
    sys.exit(1)

if mse > 1.0:
    print("MSE too high. Failing pipeline.")
    sys.exit(1)

print(" Model evaluation passed.")
