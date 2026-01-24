import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

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
# Accuracy
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Loan Risk Prediction")
plt.show()

# -----------------------------
# Feature Importance
# -----------------------------
importances = model.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importances)
plt.title("Feature Importance")
plt.show()

# -----------------------------
# Predict New Person
# -----------------------------
new_person = pd.DataFrame([{
    "Age": 35,
    "Income": 60000,
    "EmploymentType": label_encoders["EmploymentType"].transform(["Salaried"])[0],
    "ResidenceType": label_encoders["ResidenceType"].transform(["Owned"])[0],
    "CreditScore": 720,
    "LoanAmount": 20000,
    "LoanTerm": 36,
    "PreviousDefault": label_encoders["PreviousDefault"].transform(["No"])[0]
}])

prediction = model.predict(new_person)[0]
risk = label_encoders["RiskCategory"].inverse_transform([prediction])[0]

print("\nPredicted Risk Category for New Customer:", risk)
