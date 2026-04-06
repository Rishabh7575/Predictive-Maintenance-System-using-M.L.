from src.preprocessing import load_cmaps_data, add_rul
from src.featureEngineering import add_rolling_features
from src.labelling import generate_labels
from src.trainBaseline import train_models

path = "data/raw/train_FD001.txt"
sensor_cols = [f"s_{i}" for i in range(1, 22)]

df = load_cmaps_data(path)
df = add_rul(df)
df = add_rolling_features(df, sensor_cols)

df = df.dropna()   # IMPORTANT  as it reduces data( but ensure fixed size considerations)

print(df.head())
print(df.shape)

df = generate_labels(df)
X = df.drop(columns=["unit_id","cycle","RUL","label"])
y = df["label"]

print("Feature shape:", X.shape)
print("Label distribution:\n", y.value_counts())

# Step: Train baseline models
results = train_models(X, y)

print("Model Results:", results)