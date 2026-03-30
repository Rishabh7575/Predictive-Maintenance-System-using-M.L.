from src.preprocessing import load_cmaps_data, add_rul
from src.featureEngineering import add_rolling_features

path = "data/raw/train_FD001.txt"
sensor_cols = [f"s_{i}" for i in range(1, 22)]

df = load_cmaps_data(path)
df = add_rul(df)
df = add_rolling_features(df, sensor_cols)

df = df.dropna()   # IMPORTANT

print(df.head())
print(df.shape)