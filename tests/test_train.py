from src.utils import load_data
from src.train import train_and_save_model
from sklearn.linear_model import LinearRegression
import joblib
import os

def test_data_loading():
    X, y = load_data()
    assert X.shape[0] == y.shape[0]

def test_model_instance():
    X, y = load_data()
    model = LinearRegression()
    model.fit(X, y)
    assert isinstance(model, LinearRegression)

def test_model_training():
    train_and_save_model()
    model = joblib.load("models/model.joblib")
    assert hasattr(model, 'coef_')

def test_r2_score_threshold():
    X, y = load_data()
    model = joblib.load("models/model.joblib")
    r2 = model.score(X, y)
    assert r2 > 0.5  
