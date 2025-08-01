# MLOps Linear Regression Pipeline

## Objective
This project implements an end-to-end MLOps pipeline for Linear Regression using the California Housing Dataset, focusing on:

- Model training & evaluation
- Manual quantization
- Docker-based inference
- CI/CD with GitHub Actions

---

## Components

- **Dataset**: `sklearn.datasets.fetch_california_housing`
- **Model**: `LinearRegression` (scikit-learn)
- **Quantization**: Manual (via NumPy)
- **Containerization**: Docker
- **CI/CD**: GitHub Actions (auto-triggered on push to `main`)

---

## Project Structure

| Step | Component       | Description                          |
|------|------------------|--------------------------------------|
| 1    | `train.py`       | Train and save model                 |
| 2    | `test_train.py`  | Unit tests for training and metrics  |
| 3    | `quantize.py`    | Manual model quantization            |
| 4    | `predict.py`     | Run sample prediction using model    |
| 5    | `Dockerfile`     | Container to run prediction          |
| 6    | `ci.yml`         | GitHub Actions workflow for CI/CD    |

---

## Docker Usage

To build and run inside a container:

```bash
docker build -t linear-reg-app .
docker run linear-reg-app
```

---

## CI/CD Pipeline

The pipeline is triggered on every push to the `main` branch. It includes:

1. ✅ Unit Testing
2. ✅ Model Training & Quantization
3. ✅ Docker Image Build & Execution

Implemented in: `.github/workflows/ci.yml`

---

## Benchmark & Evaluation

This project uses the California Housing dataset to predict median house values using `LinearRegression` from scikit-learn. The target variable is house price (in $100,000).

### Model Evaluation Metrics

| Metric     | Expected Range | Actual (example run) | Interpretation |
|------------|----------------|----------------------|----------------|
| R² Score   | 0.60 – 0.70    | (e.g., 0.65)         | Acceptable linear fit |
| MSE (Loss) | < 1.0          | (e.g., 0.53)         | Realistic range (~$53k) |

These values are considered good benchmarks for baseline regression using no feature engineering.

### Reference Ranges

| Model                    | R² Score | MSE   |
|--------------------------|----------|-------|
| Linear Regression        | 0.60–0.68| 0.5–0.8 |
| Random Forest Regressor  | ~0.82    | ~0.2  |
| Gradient Boosting (XGB)  | ~0.85    | ~0.18 |

---

**Conclusion:**  
The current implementation produces results consistent with published benchmarks. Improvements are possible using advanced models and feature engineering, but the focus here is on end-to-end MLOps pipeline correctness.

---