import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("../data/loan_risk_data.csv")

# -----------------------------
# Encode Categorical Columns
# -----------------------------
label_encoders = {}

for col in ["EmploymentType", "ResidenceType", "PreviousDefault", "RiskCategory"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("RiskCategory", axis=1)
y = df["RiskCategory"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)

# -----------------------------
# Save for Visualization
# -----------------------------
joblib.dump(model, "../artifacts/loan_model.pkl")
joblib.dump(cm, "../artifacts/confusion_matrix.pkl")
joblib.dump(X.columns.tolist(), "../artifacts/features.pkl")


print("Model and results saved.")
