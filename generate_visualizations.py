import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import load_cmaps_data, add_rul
from src.featureEngineering import add_rolling_features
from src.labelling import generate_labels
from src.trainBaseline import train_models
from src.train_xgboost import train_xgboost
from src.evaluation import evaluate_model

# Create visualizations directory
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

print("--- Loading Data ---")
path = "data/raw/train_FD001.txt"
sensor_cols = [f"s{i}" for i in range(1, 22)]

df = load_cmaps_data(path)
df = add_rul(df)

# 4. RUL Distribution Histogram
print("Generating RUL Distribution Histogram...")
plt.figure(figsize=(10, 6))
sns.histplot(df['RUL'], bins=10, kde=True, color='skyblue')
plt.title('Remaining Useful Life (RUL) Distribution')
plt.xlabel('RUL (Cycles)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig(os.path.join(output_dir, 'rul_distribution_histogram.png'), bbox_inches='tight')
plt.close()

print("--- Preprocessing ---")
df = add_rolling_features(df, sensor_cols)
rolling_cols = []
for s in sensor_cols:
    rolling_cols += [f"{s}_mean", f"{s}_std", f"{s}_trend"]

df = df.dropna(subset=rolling_cols).reset_index(drop=True)
df = generate_labels(df)

# 1. Class Distribution Pie Chart
print("Generating Class Distribution Pie Chart...")
class_counts = df['label'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=['Normal (0)', 'Failure Imminent (1)'], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
plt.title('Class Distribution')
plt.savefig(os.path.join(output_dir, 'class_distribution_pie.png'), bbox_inches='tight')
plt.close()

X = df.drop(columns=["unit_id","cycle","RUL","label"])
y = df["label"]

# 2. Model Performance Comparison
print("Generating Model Performance Bar Chart...")
results = train_models(X, y)
xgb_model, xgb_f1, X_test, y_test = train_xgboost(X, y)

models = ['Logistic Regression', 'Random Forest', 'XGBoost']
f1_scores = [results['logistic_f1'], results['random_forest_f1'], xgb_f1]

plt.figure(figsize=(8, 6))
bars = plt.bar(models, f1_scores, color=['#4daf4a', '#377eb8', '#ff7f00'])
plt.title('Model Performance Comparison (F1 Score)')
plt.ylabel('F1 Score')
plt.ylim(0, 1.0)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom', fontweight='bold')
plt.savefig(os.path.join(output_dir, 'model_performance_bar.png'), bbox_inches='tight')
plt.close()

# 3. Confusion Matrix (XGBoost)
print("Generating Confusion Matrix Heatmap...")
evaluation = evaluate_model(xgb_model, X_test, y_test)
cm = evaluation["confusion_matrix"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix (XGBoost)')
plt.savefig(os.path.join(output_dir, 'confusion_matrix_heatmap.png'), bbox_inches='tight')
plt.close()

# 5. Feature Importance
print("Generating Feature Importance Charts...")
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=xgb_importances.values, y=xgb_importances.index, palette='viridis')
plt.title('Top 10 Feature Importances (XGBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.savefig(os.path.join(output_dir, 'feature_importance_xgb_bar.png'), bbox_inches='tight')
plt.close()

rf_model = joblib.load("models/rf_model.pkl")
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=rf_importances.values, y=rf_importances.index, palette='magma')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.savefig(os.path.join(output_dir, 'feature_importance_rf_bar.png'), bbox_inches='tight')
plt.close()

print(f"All visualizations have been successfully saved to the '{os.path.abspath(output_dir)}' directory.")
