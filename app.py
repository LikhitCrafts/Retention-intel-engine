import time
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Load model once at startup (avoids disk I/O latency on every API request)
model = joblib.load("retention_model.joblib")

app = FastAPI(
    title="Enterprise Customer Retention & Drift-Aware Inference Engine",
    description="Production-ready ML service predicting customer churn risk and monitoring input drift.",
    version="1.0.0"
)

# 2. Strict Schema Validation (Ensures data compatibility & reliability)
class CustomerProfile(BaseModel):
    tenure: int = Field(..., ge=0, le=120, description="Months with the company")
    MonthlyCharges: float = Field(..., gt=0, le=500, description="Monthly bill amount")
    TotalCharges: float = Field(..., ge=0, description="Total amount billed so far")
    Contract: Literal["Month-to-month", "One year", "Two year"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    TechSupport: Literal["Yes", "No", "No internet service"]

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict_retention_risk(customer: CustomerProfile):
    start_time = time.perf_counter()
    try:
        # Convert validated input to a 1-row pandas DataFrame
        input_df = pd.DataFrame([customer.model_dump()])

        # Run model inference
        churn_prob = float(model.predict_proba(input_df)[0][1])
        churn_pred = int(churn_prob >= 0.5)

        # Basic data drift / out-of-distribution check
        drift_warning = None
        if customer.MonthlyCharges > 120.0:
            drift_warning = "MonthlyCharges exceeds 99th percentile of training distribution."

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "churn_prediction": churn_pred,
            "churn_probability": round(churn_prob, 4),
            "risk_level": "High" if churn_prob >= 0.65 else ("Medium" if churn_prob >= 0.35 else "Low"),
            "inference_latency_ms": latency_ms,
            "drift_warning": drift_warning
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")