import sys
import os
sys.path.append(os.path.abspath("src"))

from preprocessing import load_cmaps_data, add_rul
from featureEngineering import add_rolling_features

path = "data/raw/train_FD001.txt"

sensor_cols = [f"s{i}" for i in range(1, 22)]

df = load_cmaps_data(path)
print(df.head())
print(df.shape)

df = add_rul(df)
print(df.head())

df = add_rolling_features(df, sensor_cols)
print(df.head())



















# df = load_cmaps_data(path)
# print(df.head())
# print(df.shape())

# df = add_rul(df)
# print(df.head())

# df = add_rolling_features(df, sensor_cols)
# print(df.head())