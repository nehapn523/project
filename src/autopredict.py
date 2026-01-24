import pandas as pd
import joblib
from datetime import datetime

# -----------------------------
# Load Trained Model
# -----------------------------
model = joblib.load("../artifacts/loan_model.pkl")

# -----------------------------
# Encoding Maps (same as training)
# -----------------------------
employment_map = {"Salaried": 0, "Self-employed": 1}
residence_map = {"Owned": 0, "Rented": 1}
default_map = {"No": 0, "Yes": 1}
risk_map = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}

# -----------------------------
# Load New Customer Data
# -----------------------------
input_path = "../data/new_customers.csv"   # <-- your new dataset
df = pd.read_csv(input_path)

print("Loaded new customer data:")
print(df.head())

# -----------------------------
# Encode Categorical Columns
# -----------------------------
df["EmploymentType"] = df["EmploymentType"].map(employment_map)
df["ResidenceType"] = df["ResidenceType"].map(residence_map)
df["PreviousDefault"] = df["PreviousDefault"].map(default_map)

# -----------------------------
# Predict Risk
# -----------------------------
predictions = model.predict(df)
df["PredictedRisk"] = [risk_map[p] for p in predictions]

# -----------------------------
# Save Output File
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"../data/predicted_new_customers_{timestamp}.csv"
df.to_csv(output_path, index=False)

print("\nPredictions saved to:")
print(output_path)

print("\nSample output:")
print(df.head())
