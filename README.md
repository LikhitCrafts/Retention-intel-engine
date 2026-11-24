# Retention Intel Engine

This project is an end-to-end Machine Learning inference service built using Python, pandas, scikit-learn, and FastAPI. It predicts customer churn probability in real time based on customer profile attributes.

The system separates offline model training from online REST API serving, ensuring reliable data validation, fast inference latency, and automated quality testing before deployment.

## Features
- Automated data cleaning, scaling, and encoding via `scikit-learn` `Pipeline`
- Imbalanced class handling using `class_weight="balanced"` (78% Churn Recall, 0.8394 ROC-AUC)
- Low-latency REST API built with FastAPI (`<15ms` inference response time)
- Strict input validation using Pydantic schemas (blocks invalid/malformed inputs with `422` errors)
- Out-of-distribution input monitoring (flags unusually high billing values as drift warnings)
- Automated functional test suite using `pytest`

## Project Structure
```text
Retention-intel-engine/
│── venv/                     # Virtual environment (ignored in Git)
│── train.py                  # Data ingestion, preprocessing, model training & serialization
│── retention_model.joblib    # Serialized ML pipeline artifact
│── app.py                    # FastAPI server, Pydantic validation & /predict endpoint
│── test_app.py               # Automated pytest functional test suite
│── requirements.txt          # Project dependencies
│── .gitignore                # Git ignore rules
│── README.md                 # Project documentation

How It Works
Offline Training (train.py)

Loads the IBM Telco Customer Churn dataset (7,043 customer records)

Cleans missing values and encodes target labels (Yes → 1, No → 0)

Splits data into stratified train (80%) and test (20%) sets

Fits a ColumnTransformer (SimpleImputer, StandardScaler, OneHotEncoder) + RandomForestClassifier pipeline

Evaluates Recall and ROC-AUC on unseen test data

Saves the complete preprocessing and model pipeline to retention_model.joblib

Live Inference (app.py)

Loads retention_model.joblib into memory once at server startup

Validates incoming JSON requests via Pydantic (CustomerProfile)

Runs prediction and returns churn probability, risk level (Low / Medium / High), latency in milliseconds, and any drift warnings

Quality Assurance (test_app.py)

Tests /health endpoint status

Verifies valid customer payloads return 200 OK with proper probability bounds

Verifies invalid inputs (e.g., negative tenure) are rejected with 422 Unprocessable Entity

Requirements
pandas

scikit-learn

fastapi

uvicorn

joblib

pytest

httpx

DOS
py -m venv venv
venv\Scripts\activate
py -m pip install --upgrade pip
pip install -r requirements.txt
How to Run
Step 1 — Train and save the ML pipeline:

DOS
python train.py
Step 2 — Run automated functional tests:

DOS
pytest -v
Step 3 — Start the live inference API server:

DOS
uvicorn app:app --reload
Open http://127.0.0.1:8000/docs in your browser to test live predictions via the interactive Swagger UI.

Notes
Keeping preprocessing inside the scikit-learn Pipeline prevents data leakage and guarantees identical transformations during training and live API inference.

Ideal for demonstrating production-ready ML deployment, API integration, and automated testing.

Future Improvements
Add Docker containerization for cloud deployment

Implement logging of live predictions to a SQLite/PostgreSQL database for historical drift tracking

Compare Random Forest against XGBoost and LightGBM with hyperparameter tuning

Add CI/CD GitHub Actions workflow to run pytest automatically on every push