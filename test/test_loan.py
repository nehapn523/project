import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/loan_risk_data.csv"
ARTIFACT_PATH = "artifacts/"

def test_dataset_exists():
    assert os.path.exists(DATA_PATH)

def test_dataset_not_empty():
    df = pd.read_csv(DATA_PATH)
    assert not df.empty

def test_model_file_created():
    assert os.path.exists(ARTIFACT_PATH + "loan_model.pkl")

def test_confusion_matrix_created():
    assert os.path.exists(ARTIFACT_PATH + "confusion_matrix.pkl")

def test_features_file_created():
    features = joblib.load(ARTIFACT_PATH + "features.pkl")
    assert isinstance(features, list)
    assert len(features) > 0

def test_model_type():
    model = joblib.load(ARTIFACT_PATH + "loan_model.pkl")
    assert isinstance(model, RandomForestClassifier)
