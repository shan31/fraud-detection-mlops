from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

# Create Fast Api instance
app = FastAPI(
    title="Fraud Detection API",
    description="API for fraud detection",
    version="0.0.1",
)
# Load the Trained Model
model = None
try:
    model = XGBClassifier()
    model.load_model("models/fraud_detection_model.json")
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Model not loaded - {e}")
    print("API will run but predictions will fail until model is provided")

# Define the Input Data Schema
class TransactionData(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
# Define the Predict API endpoint
@app.post("/predict")
def predict(data: TransactionData):
    if model is None:
        return {"error": "Model not loaded", "status": 503}
    input_data = pd.DataFrame([data.dict()])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probability of fraud
    return {
        "prediction": int(prediction),
        "is_fraud": bool(prediction),
        "fraud_probability": round(float(probability), 4)
    }
# Define the Health Check API endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "model": "XGBoost Fraud Detection",
        "model_loaded": model is not None
    }
