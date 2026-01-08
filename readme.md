# 🔐 Credit Card Fraud Detection - MLOps Project

A production-ready **Credit Card Fraud Detection** system with end-to-end MLOps pipeline.

![Python](https://img.shields.io/badge/Python-3.9-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Azure](https://img.shields.io/badge/Azure-Deployed-blue)

## 🎯 Live Demo

**API Endpoint:** http://fraud-api-shan.eastus.azurecontainer.io:8000

**Swagger Docs:** http://fraud-api-shan.eastus.azurecontainer.io:8000/docs

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Precision | 97.9% |
| Recall | 100% |
| F1-Score | 98.9% |
| AUC-ROC | 99.99% |

---

## 🛠️ Tech Stack

- **ML Model:** XGBoost
- **API:** FastAPI + Uvicorn
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Cloud:** Azure (ACR + ACI)
- **Experiment Tracking:** MLflow
- **Testing:** Pytest

---

## 🚀 Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/shan31/fraud-detection-mlops.git
cd fraud-detection-mlops

# Create environment
conda create -n fraud-detection python=3.9 -y
conda activate fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run API
uvicorn api.app:app --reload
```

### Docker

```bash
# Build image
docker build -t fraud-detection-api .

# Run container
docker run -v ./models:/app/models -p 8000:8000 fraud-detection-api
```

---

## 📁 Project Structure

```
fraud-detection/
├── api/app.py              # FastAPI endpoints
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── drift_detection.py
├── models/                 # Trained models
├── configs/config.yaml     # Configuration
├── tests/test_api.py       # Unit tests
├── Dockerfile
└── requirements.txt
```

---

## 🔌 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Predict Fraud
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Time":0,"V1":-1.35,...,"Amount":149.62}'
```

### Response
```json
{
  "prediction": 0,
  "is_fraud": false,
  "fraud_probability": 0.0023
}
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

---

## 📈 Features

- ✅ SMOTE for class imbalance
- ✅ MLflow experiment tracking
- ✅ FastAPI REST endpoints
- ✅ Docker containerization
- ✅ GitHub Actions CI/CD
- ✅ Azure cloud deployment
- ✅ Drift detection with KS Test
- ✅ Prediction logging

---

## 📄 License

MIT License
