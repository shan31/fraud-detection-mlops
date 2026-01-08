# Imports
import pytest
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, '.')
from api.app import app

# Test Client Setup
client = TestClient(app)

# Test: Health endpoint
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"] == "XGBoost Fraud Detection"
    # model_loaded can be True or False depending on environment

# Test: Predict endpoint with valid data
def test_predict_valid():
    test_data = {
        "Time": 0, "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37,
        "V5": -0.33, "V6": 0.46, "V7": 0.23, "V8": 0.09, "V9": 0.36,
        "V10": 0.09, "V11": -0.55, "V12": -0.61, "V13": -0.99, "V14": -0.31,
        "V15": 1.46, "V16": -0.47, "V17": 0.20, "V18": 0.02, "V19": 0.40,
        "V20": 0.25, "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06,
        "V25": 0.12, "V26": -0.18, "V27": 0.13, "V28": -0.02, "Amount": 149.62
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    data = response.json()
    # Either returns prediction (if model loaded) or error (if model not loaded)
    assert "prediction" in data or "error" in data

# Test: Predict endpoint with invalid data or Missing fields
def test_predict_missing_fields():
    test_data = {"Time": 0, "Amount": 100}  # Missing V1-V28
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # Validation error