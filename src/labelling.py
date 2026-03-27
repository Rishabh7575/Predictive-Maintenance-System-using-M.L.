def generate_labels(df, threshold=30):
    df = df.copy()
    df["label"] = (df["RUL"] <= threshold).astype(int)
    return df