import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
}

h1, h2, h3 {
    color: white;
}

[data-testid="stMetricValue"] {
    color: #22c55e;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Dashboard")

    st.markdown("""
    ### Project Information

    **Dataset**
    - NASA CMAPSS

    **Models**
    - Random Forest
    - XGBoost

    **Goal**
    - Predict equipment failure
    """)

# ---------------- TITLE ----------------
st.title("🔧 Predictive Maintenance Dashboard")

st.markdown(
    "Upload sensor data and predict equipment failure probability."
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/xgb_model.pkl")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload Sensor CSV",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data")

    st.dataframe(df.head())

    try:

        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]

        df["Prediction"] = preds
        df["Failure_Probability"] = probs

        # ---------- METRICS ----------
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Records",
            len(df)
        )

        col2.metric(
            "Predicted Failures",
            int(preds.sum())
        )

        col3.metric(
            "Failure Rate",
            f"{100 * preds.mean():.2f}%"
        )

        # ---------- PROGRESS ----------
        st.subheader("⚠️ Average Failure Probability")

        st.progress(float(probs.mean()))

        # ---------- RESULTS ----------
        st.subheader("📈 Predictions")

        st.dataframe(df.head())

        # ---------- CHART ----------
        st.subheader("📉 Failure Probability Distribution")

        chart_df = pd.DataFrame({
            "Failure Probability": probs
        })

        st.bar_chart(chart_df)

        # ---------- DOWNLOAD ----------
        csv = df.to_csv(index=False)

        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

        st.success("Prediction completed successfully!")

    except Exception as e:
        st.error(f"Error: {e}")