from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

# 1. Load the data
X, y = fetch_california_housing(return_X_y=True)

# 2. Simulate quantization by converting to float16
X_quantized = X.astype(np.float16)

# 3. Load the trained model
model = joblib.load("models/model.joblib")

# 4. Predict on quantized input
predictions = model.predict(X_quantized)

# 5. Print first few predictions as "dequantized" output
print("Dequantized prediction sample:", np.round(predictions[:5], 3))

# 6. Save the model again as "quantized" (simulated)
joblib.dump(model, "models/model_quantized.joblib")
