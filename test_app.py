from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_valid_churn_prediction():
    payload = {
        "tenure": 2,
        "MonthlyCharges": 85.5,
        "TotalCharges": 171.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "TechSupport": "No"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["Low", "Medium", "High"]
    assert "inference_latency_ms" in data

def test_invalid_input_rejected():
    bad_payload = {
        "tenure": -5,
        "MonthlyCharges": 85.5,
        "TotalCharges": 171.0,
        "Contract": "InvalidContract",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "TechSupport": "No"
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422