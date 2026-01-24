import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("../artifacts/loan_model.pkl")

# Encoding maps
employment_map = {"Salaried": 0, "Self-employed": 1}
residence_map = {"Owned": 0, "Rented": 1}
default_map = {"No": 0, "Yes": 1}
risk_map = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Batch Loan Risk Predictor", layout="centered")
st.title("📂 Batch Loan Risk Prediction Dashboard")
st.write("Upload CSV file to predict loan risk for all customers.")

uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    try:
        # Drop target if present
        if "RiskCategory" in df.columns:
            df = df.drop("RiskCategory", axis=1)

        # Encode
        df["EmploymentType"] = df["EmploymentType"].map(employment_map)
        df["ResidenceType"] = df["ResidenceType"].map(residence_map)
        df["PreviousDefault"] = df["PreviousDefault"].map(default_map)

        # Predict
        predictions = model.predict(df)
        df["PredictedRisk"] = [risk_map[p] for p in predictions]

        st.subheader("Prediction Results")
        st.dataframe(df.head())

        # -----------------------------
        # Risk Count
        # -----------------------------
        st.subheader("Risk Distribution Count")
        risk_counts = df["PredictedRisk"].value_counts()
        st.write(risk_counts)

        # -----------------------------
        # Pie Chart
        # -----------------------------
        st.subheader("Risk Distribution Chart")

        fig, ax = plt.subplots()
        ax.pie(
            risk_counts,
            labels=risk_counts.index,
            autopct="%1.1f%%",
            startangle=90
        )
        ax.axis("equal")
        st.pyplot(fig)

        # -----------------------------
        # Save All Predictions
        # -----------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_path = f"../data/predicted_results_{timestamp}.csv"
        df.to_csv(full_path, index=False)

        # -----------------------------
        # Save Only High Risk
        # -----------------------------
        high_risk_df = df[df["PredictedRisk"] == "High Risk"]
        high_risk_path = f"../data/high_risk_customers_{timestamp}.csv"
        high_risk_df.to_csv(high_risk_path, index=False)

        st.success("Prediction files saved in data folder!")

        # -----------------------------
        # Download Buttons
        # -----------------------------
        st.download_button(
            "Download All Predictions",
            data=df.to_csv(index=False),
            file_name="all_predictions.csv",
            mime="text/csv"
        )

        st.download_button(
            "Download High Risk Customers",
            data=high_risk_df.to_csv(index=False),
            file_name="high_risk_customers.csv",
            mime="text/csv"
        )

        st.info("Files also saved automatically in data folder.")

    except Exception as e:
        st.error("Error processing file. Please check CSV format.")
        st.text(str(e))
