import joblib
import pandas as pd
import numpy as np
from src.preprocessing import load_cmaps_data, add_rul
from src.featureEngineering import add_rolling_features
from src.labelling import generate_labels
from src.trainBaseline import train_models
from src.train_xgboost import train_xgboost
from src.evaluation import evaluate_model

print("--- Data Loading and Preprocessing ---")
path = "data/raw/train_FD001.txt"
sensor_cols = [f"s{i}" for i in range(1, 22)]

df = load_cmaps_data(path)
df = add_rul(df)

print("\n--- 4. RUL Distribution Histogram ---")
rul_hist, bins = np.histogram(df["RUL"], bins=10)
print("Bins:", bins)
print("Counts:", rul_hist)

df = add_rolling_features(df, sensor_cols)
rolling_cols = []
for s in sensor_cols:
    rolling_cols += [f"{s}_mean", f"{s}_std", f"{s}_trend"]

df = df.dropna(subset=rolling_cols).reset_index(drop=True)
df = generate_labels(df)

print("\n--- 1. Class Distribution Pie Chart ---")
print(df["label"].value_counts())

X = df.drop(columns=["unit_id","cycle","RUL","label"])
y = df["label"]

print("\n--- 2. Model Performance Comparison ---")
results = train_models(X, y)
print("Baseline Results:", results)

xgb_model, xgb_f1, X_test, y_test = train_xgboost(X, y)
print("XGBoost F1:", xgb_f1)

evaluation = evaluate_model(xgb_model, X_test, y_test)
print("\n--- 3. Confusion Matrix (XGBoost) ---")
print(evaluation["confusion_matrix"])

print("\n--- 5. Feature Importance ---")
print("\nXGBoost Feature Importances (Top 10):")
xgb_importances = xgb_model.feature_importances_
xgb_feat_imp = pd.Series(xgb_importances, index=X.columns).sort_values(ascending=False)
print(xgb_feat_imp.head(10))

print("\nRandom Forest Feature Importances (Top 10):")
rf_model = joblib.load("models/rf_model.pkl")
rf_importances = rf_model.feature_importances_
rf_feat_imp = pd.Series(rf_importances, index=X.columns).sort_values(ascending=False)
print(rf_feat_imp.head(10))
