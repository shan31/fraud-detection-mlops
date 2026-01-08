import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os

# Load config.yaml
with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)
# Read raw csv
df = pd.read_csv(config["data"]["raw"])
print(df.head())
print(df.shape)
# Separate features (X) and target (y)
X = df.drop('Class', axis=1)
y = df['Class']
print(f"Features shape: {X.shape}, Target shape: {y.shape}")
# Apply smote
print("Before Smote:")
print(y.value_counts())
smote = SMOTE(random_state=config["model"]["random_state"])
X_resampled, y_resampled = smote.fit_resample(X, y)
print("After SMOTE:", pd.Series(y_resampled).value_counts())
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=config["model"]["test_size"], random_state=config["model"]["random_state"])
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
# Save to processed folder
os.makedirs(config["data"]["processed"], exist_ok=True)
print("Saving processed data to:", config["data"]["processed"])

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv(os.path.join(config["data"]["processed"], "train.csv"), index=False)
test_df.to_csv(os.path.join(config["data"]["processed"], "test.csv"), index=False)