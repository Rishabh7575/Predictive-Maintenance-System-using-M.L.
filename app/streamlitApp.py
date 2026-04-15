import streamlit as st
import pandas as pd
import joblib

st.title("🔧 Predictive Maintenance System")

# Load model
model = joblib.load("models/xgb_model.pkl")

uploaded_file = st.file_uploader("Upload Sensor CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Input Data")
    st.dataframe(df.head())

    try:
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]

#         df["Prediction"] = preds
#         df["Failure_Probability"] = probs

#         st.subheader("📈 Predictions")
#         st.dataframe(df.head())

#         st.success("Prediction completed successfully!")

#     except Exception as e:
#         st.error(f"Error: {e}")