# MLOps Linear Regression Pipeline

## Objective
This project implements an end-to-end MLOps pipeline for Linear Regression using the California Housing Dataset, focusing on:

- Model training & evaluation
- Manual quantization
- Docker-based inference
- CI/CD with GitHub Actions
- Deployment to DockerHub

---

## Components

- **Dataset**: `sklearn.datasets.fetch_california_housing`
- **Model**: `LinearRegression` (scikit-learn)
- **Quantization**: Manual (via NumPy `float16`)
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **CD**: DockerHub image push via secrets

---

## Project Structure

| Step | Component       | Description                          |
|------|------------------|--------------------------------------|
| 1    | `train.py`       | Train and save model                 |
| 2    | `test_train.py`  | Unit tests for training and metrics  |
| 3    | `quantize.py`    | Manual model quantization (`float16`)|
| 4    | `predict.py`     | Run full dataset prediction + summary|
| 5    | `Dockerfile`     | Container to run prediction          |
| 6    | `ci.yml`         | GitHub Actions CI/CD workflow        |
| 7    | `reporting.py`   | Saves visual metric plots            |
| 8    | `testing.py`     | Validates R² & MSE thresholds         |

---

## Docker Usage

To build and run inside a container:

```bash
docker build -t linear-reg-app .
docker run linear-reg-app
To persist predictions locally:

bash
Copy
Edit
docker run --rm -v "$(pwd)/models:/app/models" linear-reg-app
CI/CD Pipeline Overview
The pipeline is defined in .github/workflows/ci.yml, and triggered on every push to main.

📦 Stages:
🧪 Setup & Unit Test

🎯 Training and Quantize

🔨 Build Container

🧪 Test Container

🚀 Deployment (DockerHub)

GitHub Actions Integration
All metrics (metrics.json, quant_metrics.json, metrics_plot.png, predictions.json) are uploaded as artifacts

Full logs shown in GitHub Actions UI

Deployment auto-pushes image to:

docker.io/piyushkumarai1066/linear-reg-app

Benchmark & Evaluation
This project uses the California Housing dataset to predict median house values using LinearRegression from scikit-learn. The target variable is house price (in $100,000).

Model Evaluation Metrics
Metric	Expected Range	Actual (example run)	Interpretation
R² Score	0.60 – 0.70	e.g., 0.6523	Acceptable linear fit
MSE (Loss)	< 1.0	e.g., 0.5317	Errors within realistic range

All metrics are automatically evaluated and stored in models/metrics.json and models/quant_metrics.json.

Sample Predictions
A summary of full-dataset predictions is logged in predict.py, and top predictions are written to models/predictions.json.

Visual Reports
The reporting.py script plots key evaluation metrics:

Output: models/metrics_plot.png

Reference Benchmarks
Model	R² Score	MSE
Linear Regression	0.60–0.68	0.5–0.8
Random Forest Regressor	~0.82	~0.2
Gradient Boosting (XGB)	~0.85	~0.18

Conclusion
This project demonstrates a full MLOps workflow including training, testing, quantization, containerization, and continuous deployment — all automated via GitHub Actions.

The setup emphasizes:

Code reproducibility

CI/CD pipeline maturity

Deployment readiness