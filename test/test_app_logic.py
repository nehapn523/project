import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "artifacts/loan_model.pkl"

employment_map = {"Salaried": 0, "Self-employed": 1}
residence_map = {"Owned": 0, "Rented": 1}
default_map = {"No": 0, "Yes": 1}
risk_map = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}

def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert isinstance(model, RandomForestClassifier)

def test_encoding_maps_work():
    assert employment_map["Salaried"] == 0
    assert residence_map["Rented"] == 1
    assert default_map["Yes"] == 1

def test_prediction_pipeline():
    model = joblib.load(MODEL_PATH)

    # dummy input dataframe (same structure as CSV)
    df = pd.DataFrame({
        "EmploymentType": ["Salaried", "Self-employed"],
        "ResidenceType": ["Owned", "Rented"],
        "PreviousDefault": ["No", "Yes"],
        "Income": [50000, 30000],
        "LoanAmount": [200000, 150000]
    })

    # encoding
    df["EmploymentType"] = df["EmploymentType"].map(employment_map)
    df["ResidenceType"] = df["ResidenceType"].map(residence_map)
    df["PreviousDefault"] = df["PreviousDefault"].map(default_map)

    preds = model.predict(df)

    assert len(preds) == len(df)

def test_risk_mapping():
    assert risk_map[0] == "High Risk"
    assert risk_map[1] == "Low Risk"
    assert risk_map[2] == "Medium Risk"
