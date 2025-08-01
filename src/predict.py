import joblib
from sklearn.datasets import fetch_california_housing

def run_prediction():

    model = joblib.load("models/model.joblib")
    

    X, _ = fetch_california_housing(return_X_y=True)


    preds = model.predict(X[:5])
    print("Sample Predictions:", preds)

if __name__ == "__main__":
    run_prediction()


