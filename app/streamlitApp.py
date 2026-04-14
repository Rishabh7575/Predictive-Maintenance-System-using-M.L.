import streamlit as st
import pandas as pd
import joblib

st.title("Predictive Maintenance System")

model = joblib.load("models/xgb_model.pkl")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Input Data", df.head())

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    df["Prediction"] = preds
    df["Failure_Probability"] = probs

    st.write("Predictions", df.head())