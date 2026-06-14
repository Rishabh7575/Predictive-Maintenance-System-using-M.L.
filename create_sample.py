import joblib

X = joblib.load("data/processed/X.pkl")
X.head(50).to_csv("sample_input.csv", index=False)