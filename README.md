# Retention Intel Engine

This project is an **end-to-end Machine Learning inference service** that predicts customer churn probability using a trained Random Forest classification model.

The system analyzes customer information to identify the likelihood of churn, categorizes customers into different risk levels, monitors inference latency, validates incoming data, and detects potentially unusual input values.

It uses a **decoupled ML architecture**, where model training and evaluation happen offline, while real-time predictions are served through a FastAPI REST API.

---

## Features

- Predicts customer churn probability in real time
- End-to-end ML pipeline using Scikit-learn
- Random Forest classification with balanced class weights
- Automated data preprocessing using ColumnTransformer
- Input validation using Pydantic
- REST API built with FastAPI
- Customer risk categorization (Low, Medium, High)
- Inference latency monitoring
- Basic out-of-distribution input detection
- Automated API testing using pytest
- Serialized model loading for efficient inference

---

## Project Structure

```text
Retention-intel-engine/
│
├── venv/                       # Python virtual environment
│
├── train.py                    # Data preprocessing, model training & evaluation
├── retention_model.joblib      # Serialized ML pipeline
├── app.py                      # FastAPI inference service
├── test_app.py                 # Automated API tests
├── requirements.txt            # Project dependencies
├── .gitignore                  # Ignored files
└── README.md                   # Project documentation
```

---

## How It Works

### 1. Model Training (Offline)

- Fetches the IBM Telco Customer Churn dataset (7,043 records).
- Cleans missing and invalid values.
- Converts churn labels into numerical values (Yes = 1, No = 0).
- Selects numerical and categorical features.
- Splits data into 80% training and 20% testing.
- Applies preprocessing using ColumnTransformer:
  - Numerical: Median imputation + StandardScaler
  - Categorical: Most frequent imputation + OneHotEncoder
- Trains a RandomForestClassifier with balanced class weights.
- Evaluates model performance.
- Saves the complete ML pipeline as `retention_model.joblib`.

### 2. Real-Time Prediction (FastAPI)

- Loads the trained model into memory.
- Accepts customer information through a REST API.
- Validates incoming data using Pydantic.
- Predicts churn probability using `predict_proba()`.
- Classifies customers into three risk categories:

| Churn Probability | Risk Level |
|---|---|
| ≥ 0.65 | High |
| ≥ 0.35 | Medium |
| < 0.35 | Low |

- Measures inference execution time in milliseconds.
- Flags unusual inputs using a basic drift-checking rule.

### 3. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks API and model health |
| POST | `/predict` | Predicts customer churn probability |

### 4. Automated Testing

Uses `pytest` and FastAPI TestClient to verify:

- Health endpoint functionality.
- Valid customer churn predictions.
- Probability and risk-level outputs.
- Invalid input rejection (HTTP 422).

---

## Model Performance

The Random Forest model was evaluated on a test dataset containing 1,409 samples.

| Metric | Result |
|---|---:|
| ROC-AUC Score | 0.8394 |
| Accuracy | 76% |
| Churn Recall | 0.78 |
| Churn Precision | 0.53 |
| Churn F1-Score | 0.63 |
| Non-Churn Recall | 0.75 |

---

## Technologies Used

1. Python 3.11
2. Pandas
3. NumPy
4. Scikit-learn
5. Joblib
6. FastAPI
7. Pydantic
8. Uvicorn
9. Pytest
10. HTTPX

---

## How to Run

### 1. Create Virtual Environment

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train.py
```

This generates the serialized model:

```text
retention_model.joblib
```

### 4. Start the API Server

```bash
uvicorn app:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Run Automated Tests

```bash
pytest -v
```

Expected result:

```text
3 passed
```

---

## Key Implementation Highlights

- **Leakage-Free Preprocessing:** Uses Scikit-learn Pipeline and ColumnTransformer to ensure preprocessing is fitted only on training data.
- **Class Imbalance Handling:** Uses `class_weight="balanced"` to improve minority-class churn detection.
- **Efficient Inference:** Loads the serialized model once instead of loading it for every prediction.
- **Schema Validation:** Rejects invalid customer inputs before model inference.
- **Monitoring:** Tracks inference latency and flags unusual input values.
- **Automated Testing:** Verifies API functionality and input validation.

---

## Future Improvements

- Implement advanced data drift detection using statistical techniques.
- Add model versioning and experiment tracking using MLflow.
- Integrate a frontend dashboard for customer churn analytics.
- Add Docker containerization and cloud deployment.
- Implement continuous monitoring and model retraining.
- Add explainable AI using SHAP feature importance.

---

**Note:** This project demonstrates a complete ML deployment workflow, covering data preprocessing, model training, evaluation, serialization, REST API development, inference monitoring, and automated testing.