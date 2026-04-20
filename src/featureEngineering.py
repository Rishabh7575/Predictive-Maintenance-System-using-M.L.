def add_rolling_features(df, sensor_cols, window=10):
    df = df.copy()
    for s in sensor_cols:
        df[f"{s}_mean"] = (
            df.groupby("unit_id")[s]
              .rolling(window)
              .mean()
              .reset_index(0, drop=True)
        )
        df[f"{s}_std"] = (
            df.groupby("unit_id")[s]
              .rolling(window)
              .std()
              .reset_index(0, drop=True)
        )
        df[f"{s}_trend"] = df.groupby("unit_id")[s].diff(window)
    return df


#data leakage will be adjusted tomorrow dated : 22/04/26
#Docs updation date : 25/04/26