import joblib
import numpy as np
import sys
from src.utils import load_data
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    X, y = load_data()
    model = joblib.load("models/model.joblib")

    preds = model.predict(X)
    r2 = r2_score(y, preds)
    mse = mean_squared_error(y, preds)

    print(f"[Evaluation] R² Score: {r2}")
    print(f"[Evaluation] MSE: {mse}")

    # Basic thresholds
    if r2 < 0.5:
        print("❌ R² score too low. Failing pipeline.")
        sys.exit(1)
    if mse > 1.0:
        print("❌ MSE too high. Failing pipeline.")
        sys.exit(1)

    # ✅ Additional 10 Test Cases

    # 1. Prediction length matches input length
    assert len(preds) == len(y), "Prediction length mismatch"

    # 2. Prediction is a 1D numpy array
    assert isinstance(preds, np.ndarray) and preds.ndim == 1, "Predictions must be 1D array"

    # 3. No NaNs or infs in predictions
    assert not np.isnan(preds).any(), "Predictions contain NaNs"
    assert not np.isinf(preds).any(), "Predictions contain Infs"

    # 4. Residuals are within sane range
    residuals = np.abs(y - preds)
    assert np.percentile(residuals, 95) < 3.0, "High residuals detected"

    # 5. Model is instance of LinearRegression
    assert isinstance(model, LinearRegression), "Model is not LinearRegression"

    # 6. Coefficients are not all zeros
    coefs = model.coef_
    assert not np.allclose(coefs, 0), "Model coefficients are all zero"

    # 7. Predictions show reasonable spread
    assert np.std(preds) > 0.2, "Predictions lack variance"

    # 8. Mean prediction falls in reasonable range
    assert 0.5 < np.mean(preds) < 5.0, "Mean prediction out of bounds"

    # 9. R² and MSE are consistent
    assert r2 > 0 and mse > 0, "Invalid R² or MSE values"

    # 10. Predicted values are positive
    assert (preds > 0).mean() > 0.95, "Too many negative predictions"

    print("✅ All evaluation checks passed.")
