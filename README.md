![CI Status](https://github.com/joseph-m-pasia/kg-default-of-credit-cards/actions/workflows/ci.yml/badge.svg)

# Credit Default Prediction API

A production-ready machine learning project that predicts whether a credit card client is likely to default on their next payment.

The project demonstrates an end-to-end ML workflow, from data preprocessing and model training to deployment as a REST API using FastAPI and Docker, with automated testing and CI/CD through GitHub Actions.

---

## Project Highlights

- End-to-end machine learning pipeline
- Feature engineering and preprocessing using Scikit-learn Pipelines
- Hyperparameter tuning with GridSearchCV
- Champion model selection
- Model serialization with Joblib
- REST API built with FastAPI
- Docker containerization
- GitHub Actions CI/CD
- Unit testing with Pytest
- Code quality checks using Flake8

---

## Problem Statement

Financial institutions need to estimate the likelihood that a customer will default on their credit card payments. This project uses historical customer information to build a classification model that predicts default risk.

Dataset:
- **Default of Credit Card Clients**
- Source: UCI Machine Learning Repository

Target variable:

- `default.payment.next.month`
    - 0 = No Default
    - 1 = Default

---

## Tech Stack

### Machine Learning

- Python
- Scikit-learn
- Pandas
- NumPy
- Joblib

### API

- FastAPI
- Uvicorn
- Pydantic

### DevOps

- Docker
- GitHub Actions

### Testing

- Pytest
- Flake8

---

## Project Structure

```text
.
├── app/
│   └── main.py                 # FastAPI application
│
├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── pkg_credit_default/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   └── utils/
│
├── tests/
│
├── Dockerfile
├── pyproject.toml
├── README.md
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Machine Learning Pipeline

The project follows a modular machine learning pipeline:

```
Raw Data
    │
    ▼
Data Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
Preprocessing Pipeline
    │
    ▼
Model Training
    │
    ▼
Hyperparameter Tuning
    │
    ▼
Champion Model Selection
    │
    ▼
Model Serialization
    │
    ▼
FastAPI Deployment
```

---

## Feature Engineering

The project includes custom feature engineering such as:

- Credit utilization ratio
- Average bill amount
- Average payment amount
- Payment delay indicators
- Missing value handling
- Feature scaling
- Categorical encoding (if required)

Custom Scikit-learn transformers are used to keep preprocessing reproducible and production-ready.

---

## Model Selection

Several classification algorithms were evaluated, including:

- Logistic Regression
- Random Forest
- Gradient Boosting (optional)
- XGBoost (optional)

Hyperparameter tuning was performed using GridSearchCV with cross-validation.

The best-performing model was selected as the champion model based on validation metrics.

---

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Example results:

| Metric | Score |
|---------|------:|
| Accuracy | 0.776 |
| Precision | 0.496 |
| Recall | 0.557 |
| F1 Score | 0.525 |
| ROC-AUC | 0.746 |

---

## Running Locally

### Clone the repository

```bash
git clone https://github.com/<username>/kg-default-of-credit-cards.git

cd credit-default-api
```

---

### Install dependencies

```bash
pip install -e .
```

or

```bash
pip install -r requirements.txt
```

---

### Start the API

```bash
uvicorn app.main:app --reload
```

API will be available at

```
http://127.0.0.1:8000
```

Interactive documentation:

```
http://127.0.0.1:8000/docs
```

---

## Example Prediction Request

```json
POST /predict

{
  "LIMIT_BAL": 20000,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 24,
  "PAY_0": 2,
  "PAY_2": 2,
  "PAY_3": -1,
  "PAY_4": -1,
  "PAY_5": -2,
  "PAY_6": -2,
  "BILL_AMT1": 3913,
  "BILL_AMT2": 3102,
  "BILL_AMT3": 689,
  "BILL_AMT4": 0,
  "BILL_AMT5": 0,
  "BILL_AMT6": 0,
  "PAY_AMT1": 0,
  "PAY_AMT2": 689,
  "PAY_AMT3": 0,
  "PAY_AMT4": 0,
  "PAY_AMT5": 0,
  "PAY_AMT6": 0
}
```

Example response

```json
{
    "prediction": 1
}
```

---

## Docker

Build the image

```bash
docker build -t credit-default-api .
```

Run the container

```bash
docker run -p 8000:8000 credit-default-api
```

---

## Running Tests

```bash
pytest
```

---

## Code Quality

Run Flake8

```bash
flake8 .
```

---

## Continuous Integration

GitHub Actions automatically performs:

- Install dependencies
- Run unit tests
- Run Flake8
- Build Docker image

This ensures every commit maintains code quality and remains deployable.

---

## Future Improvements

- SHAP model explainability
- MLflow experiment tracking
- Model monitoring
- Batch prediction endpoint
- Authentication
- Kubernetes deployment
- Cloud deployment on Azure or AWS

---

## Skills Demonstrated

This project showcases experience in:

- Machine Learning
- Feature Engineering
- Scikit-learn Pipelines
- Python Software Engineering
- REST API Development
- Docker
- FastAPI
- CI/CD
- Unit Testing
- Clean Code
- MLOps Fundamentals

---

## License

This project is intended for educational and portfolio purposes.
