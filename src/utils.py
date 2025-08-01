from sklearn.datasets import fetch_california_housing

def load_data():
    return fetch_california_housing(return_X_y=True)
