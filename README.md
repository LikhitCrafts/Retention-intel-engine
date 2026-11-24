# Enterprise Customer Retention & Drift-Aware Inference Engine

An end-to-end Machine Learning inference service built with **Python, pandas, scikit-learn, and FastAPI** to predict customer churn risk in real time while monitoring latency and input distribution drift.

## Architecture & Key Engineering Choices
* **Leakage-Free Preprocessing (`train.py`):** Uses `scikit-learn`'s `ColumnTransformer` and `Pipeline` (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) to prevent training-serving skew and data leakage.
* **Imbalanced Class Optimization:** Configured `RandomForestClassifier(class_weight="balanced")` to prioritize **Recall (78%)** on churning customers (`ROC-AUC: 0.8394`).
* **Low-Latency REST API (`app.py`):** Loads the serialized `joblib` artifact once at startup and enforces strict input validation via **Pydantic** (`Literal` types and range bounds), achieving `<15ms` inference latency.
* **Automated Quality Assurance (`test_app.py`):** Functional unit tests built with `pytest` and `TestClient` verifying health checks, valid predictions, and `422` schema rejection on malformed inputs.

## Quickstart
```bash
pip install -r requirements.txt
python train.py
uvicorn app:app --reload