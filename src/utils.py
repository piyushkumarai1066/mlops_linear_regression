from sklearn.datasets import fetch_california_housing
import numpy as np

def load_data():
    X, y = fetch_california_housing(return_X_y=True)
    reps = 5  # 20,640 × 5 = 103,200 samples
    X_large = np.tile(X, (reps, 1))
    y_large = np.tile(y, reps)
    return X_large, y_large

