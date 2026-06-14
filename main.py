import joblib
from src.preprocessing import load_cmaps_data, add_rul
from src.featureEngineering import add_rolling_features
from src.labelling import generate_labels
from src.trainBaseline import train_models
from src.train_xgboost import train_xgboost
from src.model_comparison import compare_models
from src.evaluation import evaluate_model

path = "data/raw/train_FD001.txt"
sensor_cols = [f"s{i}" for i in range(1, 22)]


df = load_cmaps_data(path)
df = add_rul(df)
df = add_rolling_features(df, sensor_cols)

rolling_cols = []
for s in sensor_cols:
    rolling_cols += [f"{s}_mean", f"{s}_std", f"{s}_trend"]

df = df.dropna(subset=rolling_cols).reset_index(drop=True)# IMPORTANT  as it reduces data( but ensure fixed size considerations)

print(df.columns)
print(df.shape)
print(df.head())

df = generate_labels(df)
X = df.drop(columns=["unit_id","cycle","RUL","label"])
y = df["label"]

print("Feature shape:", X.shape)
print("Label distribution:\n", y.value_counts())
joblib.dump(X, "data/processed/X.pkl")
joblib.dump(y, "data/processed/y.pkl")

print(X.shape)
print(y.shape)
print(y.value_counts())

# Step: Train baseline models
results = train_models(X, y)

print("Model Results:", results)

xgb_model, xgb_f1, X_test, y_test = train_xgboost(X, y)

evaluation = evaluate_model(
    xgb_model,
    X_test,
    y_test
)

print("\n--- Evaluation Metrics ---")
print("Accuracy :", round(evaluation["accuracy"], 4))
print("Precision:", round(evaluation["precision"], 4))
print("Recall   :", round(evaluation["recall"], 4))
print("F1 Score :", round(evaluation["f1"], 4))

print("\nConfusion Matrix")
print(evaluation["confusion_matrix"])

print("\nClassification Report")
print(evaluation["report"])

print("Random Forest model saved")
print("XGBoost model saved")

compare_models(results, xgb_f1)
