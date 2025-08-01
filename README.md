# MLOps Linear Regression Pipeline

## Objective
This project implements an end-to-end MLOps pipeline for Linear Regression using the California Housing Dataset.

## Components
- Dataset: `sklearn.datasets.fetch_california_housing`
- Model: `LinearRegression` (scikit-learn)
- Quantization: Manual quantization using NumPy
- Dockerized Inference
- CI/CD using GitHub Actions

## Project Structure

| Step | Component | Description |
|------|-----------|-------------|
| 1 | train.py | Train and save model |
| 2 | test_train.py | Unit tests |
| 3 | quantize.py | Manual model quantization |
| 4 | predict.py | Sample prediction |
| 5 | Dockerfile | Container for prediction |
| 6 | CI/CD | GitHub Actions Workflow |

## Docker
```bash
docker build -t linear-reg-app .
docker run linear-reg-app
