import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load Saved Artifacts
# -----------------------------
cm = joblib.load("../artifacts/confusion_matrix.pkl")
features = joblib.load("../artifacts/features.pkl")
model = joblib.load("../artifacts/loan_model.pkl")

# -----------------------------
# Confusion Matrix Visualization
# -----------------------------
plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix - Loan Risk Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j], ha="center", va="center", color="black")

plt.colorbar()
plt.show()

# -----------------------------
# Feature Importance Visualization
# -----------------------------
importances = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance in Loan Risk Prediction")
plt.show()
