import pandas as pd

def load_cmaps_data(path):
    cols=["unit_id","cycle"] + \
    [f"os{i}" for i in range(1,4)] + \
    [f"s{i}" for i in range(1,22)]

    df=pd.read_csv(path, sep=" ", header=None, names=cols)
    df=df.iloc[:,:len(cols)]        # Remove last mpt column
    df.columns=cols
    return df

def add_rul(df):
    max_cycle=df.groupby("unit_id")["cycle"].max().reset_index()
    max_cycle.columns=["unit_id","max_cycle"]

    df=df.merge(max_cycle, on="unit_id")    # Merge max cycle info
    df["RUL"] = df["max_cycle"]-df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df