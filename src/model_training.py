import pandas as pd
import yaml
try:
    import mlflow
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("⚠️ MLflow not installed - will skip experiment tracking")

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
import os

# Load config.yaml
with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load processed train/test data
train_df = pd.read_csv(config["data"]["processed"] + "/train.csv")
test_df = pd.read_csv(config["data"]["processed"] + "/test.csv")
print(train_df.head())
print(train_df.shape)
print(test_df.head())
print(test_df.shape)

# Start MLFLOW experiment
if MLFLOW_AVAILABLE:
    mlflow.set_experiment("fraud-detection-experiment")
    mlflow.start_run()

    # Log config Parameters
    mlflow.log_param("algorithm", config["training"]["algorithm"])
    mlflow.log_param("n_estimators", config["training"]["n_estimators"])
    mlflow.log_param("max_depth", config["training"]["max_depth"])
    mlflow.log_param("learning_rate", config["training"]["learning_rate"])
    mlflow.log_param("scale_pos_weight", config["training"]["scale_pos_weight"])
    mlflow.log_param("random_state", config["model"]["random_state"])

# Train XGboost model with scale_pos_weight
model = XGBClassifier(
    n_estimators=config["training"]["n_estimators"],
    max_depth=config["training"]["max_depth"],
    learning_rate=config["training"]["learning_rate"],
    scale_pos_weight=config["training"]["scale_pos_weight"],
    random_state=config["model"]["random_state"],
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(train_df.drop("Class", axis=1), train_df["Class"])

# Make predictions
y_pred = model.predict(test_df.drop("Class", axis=1))
y_pred_proba = model.predict_proba(test_df.drop("Class", axis=1))[:, 1]

# Calculate metrics (Precision, Recall, F1, F2, AUC-ROC)
precision = precision_score(test_df["Class"], y_pred)
recall = recall_score(test_df["Class"], y_pred)
f1 = f1_score(test_df["Class"], y_pred)
f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
auc_roc = roc_auc_score(test_df["Class"], y_pred_proba)

print(f"\nModel Performance Metrics:")
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)
print("F2: ", f2)
print("AUC-ROC: ", auc_roc)

# Log params, metrics, and model to MLflow
if MLFLOW_AVAILABLE:
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("f2_score", f2)
    mlflow.log_metric("auc_roc", auc_roc)
    mlflow.xgboost.log_model(model, "model")

# Save model locally
os.makedirs("models", exist_ok=True)
model.save_model("models/fraud_detection_model.json")
print("[OK] MLflow run completed!")

if MLFLOW_AVAILABLE:
    mlflow.end_run()
    print("[OK] MLflow run completed!")
else:
    print("[OK] Training completed (MLflow skipped)")


