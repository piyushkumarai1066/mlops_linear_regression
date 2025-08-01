from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np


X, y = fetch_california_housing(return_X_y=True)


X_quantized = X.astype(np.float16)


model = joblib.load("models/model.joblib")


predictions = model.predict(X_quantized)


print("Dequantized prediction sample:", np.round(predictions[:5], 3))


joblib.dump(model, "models/model_quantized.joblib")
