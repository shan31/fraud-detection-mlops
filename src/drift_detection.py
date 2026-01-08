import pandas as pd
import json
import yaml
from scipy.stats import ks_2samp
import os

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def check_drift():
    # 1. Load Training Data (Baseline)
    print("📉 Loading baseline data...")
    train_df = pd.read_csv(config["data"]["processed"] + "/train.csv")
    
    # 2. Load Production Logs
    log_path = "logs/predictions.json"
    if not os.path.exists(log_path):
        print("⚠️ No production logs found. Cannot check for drift.")
        return

    print("📈 Loading production logs...")
    data = []
    with open(log_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    prod_df = pd.DataFrame(data)

    # Implemenet Sliding Window
    window_size = 1000
    if len(prod_df) > window_size:
        prod_df = prod_df.tail(window_size)
        print(f"🪟 Using Sliding Window: Last {window_size} records")
    else:
        print(f"ℹ️ Using all {len(prod_df)} records (Accumulating data for window)")
    
    # 3. Compare Distributions (KS Test)
    drift_report = {}
    drift_detected = False
    
    print("\n🔍 Checking for Data Drift (KS Test):")
    print(f"{'Feature':<10} | {'P-Value':<10} | {'Status'}")
    print("-" * 35)
    
    # Check V1-V28 and Amount
    features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    
    for feature in features:
        if feature not in prod_df.columns:
            continue
            
        # Run KS Test
        statistic, p_value = ks_2samp(train_df[feature], prod_df[feature])
        
        # Threshold: p < 0.05 means distributions are DIFFERENT (Drift)
        is_drift = p_value < 0.05
        status = "🔴 DRIFT" if is_drift else "✅ Stable"
        
        if is_drift:
            drift_detected = True
            
        print(f"{feature:<10} | {p_value:.4f}     | {status}")
        drift_report[feature] = p_value

    if drift_detected:
        print("\n⚠️ ALERT: Data Drift Detected! Model retraining recommended.")
    else:
        print("\n✅ System Healthy: No significant drift detected.")

if __name__ == "__main__":
    check_drift()
