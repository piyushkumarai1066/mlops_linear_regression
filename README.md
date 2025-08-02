🏡 California Housing - MLOps Linear Regression Pipeline
🎯 Objective
This project implements a complete MLOps pipeline focused on Linear Regression using the California Housing Dataset, covering:

✅ Data loading & preprocessing

✅ Model training & evaluation

✅ Manual model quantization (float16)

✅ End-to-end testing & QA

✅ Containerization via Docker

✅ CI/CD using GitHub Actions

✅ Deployment to DockerHub

📁 Components Overview
Step	File/Folder	Description
1	src/train.py	Trains the model and saves it using joblib
2	tests/test_train.py	Unit tests for training step
3	src/quantize.py	Quantizes model using NumPy (float16)
4	src/predict.py	Runs full dataset inference and saves output
5	Dockerfile	Containerizes the model for portable deployment
6	.github/workflows/ci.yml	CI/CD pipeline triggered on push to main
7	reporting.py	Generates metrics_plot.png from evaluation results
8	tests/testing.py	QA tests: R² & MSE thresholds, model checks
9	tests/uat_tests.py	Final UAT validation across 20 checks

📦 Docker Usage
To run locally:

bash
Copy
Edit
docker build -t linear-reg-app .
docker run linear-reg-app
To persist predictions:

bash
Copy
Edit
docker run --rm -v "$(pwd)/models:/app/models" linear-reg-app
🚀 CI/CD Pipeline Overview
The pipeline is defined in .github/workflows/ci.yml and is triggered on:

Every push to the main branch

Manual trigger via GitHub UI

🔄 Stages:
Stage Name	Description
🧪 Setup & Unit Test	Installs deps, runs unit tests (pytest)
🎯 Training and Quantize	Trains model and performs quantization
✅ QA	Evaluates metrics, generates visual plots
🔨 Build Container	Builds the Docker image
🧪 Test Container	Runs the Docker container for prediction
🧪 UAT	Performs 20 automated checks on artifacts
🚀 Deployment	Pushes image to DockerHub on success

✅ Artifacts:
models/model.joblib

models/model_quantized.joblib

models/metrics.json

models/quant_metrics.json

models/predictions.json

models/metrics_plot.png

All available via GitHub Actions → Artifacts tab.

📊 Benchmark & Evaluation
The model predicts median house values (in $100,000s) from the California Housing dataset using LinearRegression.

📈 Evaluation Metrics
Metric	Expected Range	Example Run	Interpretation
R² Score	0.60 – 0.70	~0.6523	Strong linear fit
MSE	< 1.0	~0.5317	Acceptable predictive loss

Evaluated automatically and logged to models/metrics.json and models/quant_metrics.json.

🔍 Sample Prediction Output
json
Copy
Edit
{
  "summary": {
    "total_samples": 20640,
    "min_prediction": 0.17,
    "max_prediction": 4.72,
    "mean_prediction": 2.07
  },
  "predictions": [...]
}
See models/predictions.json for the complete output.

📉 Visual Report
The script reporting.py auto-generates a plot:

File: models/metrics_plot.png

Metrics: Bar chart showing R² and MSE values

📌 Reference Benchmarks
Model	R² Score	MSE
Linear Regression	0.60–0.68	0.5–0.8
Random Forest Regressor	~0.82	~0.2
Gradient Boosting (XGB)	~0.85	~0.18

This project hits baseline targets using Linear Regression alone — without feature engineering or complex tuning — to highlight CI/CD and MLOps best practices.

✅ Final Notes
🔁 Fully reproducible pipeline using joblib, float16, and no random seeds

🧪 20+ UAT assertions validate every artifact and metric

🐳 Deployed Docker image available at:

bash
Copy
Edit
docker pull docker.io/piyushkumarai1066/linear-reg-app:latest
🙌 Author
Piyush Kumar
Roll No: G24AI1066
Indian Institute of Technology, Jodhpur