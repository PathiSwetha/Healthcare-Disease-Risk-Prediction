import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# -----------------------------
# Create models folder
# -----------------------------
os.makedirs("models", exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/heart.csv")

print("Dataset Loaded Successfully")
print(df.head())

# -----------------------------
# Data Cleaning
# -----------------------------
print("\nChecking Missing Values...")
print(df.isnull().sum())

print("\nRemoving Duplicates...")
df.drop_duplicates(inplace=True)

# -----------------------------
# Features & Target
# -----------------------------
X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Logistic Regression
# -----------------------------
log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)

print("\n========== Logistic Regression ==========")
print("Accuracy :", round(accuracy_score(y_test, log_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, log_pred) * 100, 2), "%")
print("Recall   :", round(recall_score(y_test, log_pred) * 100, 2), "%")
print("F1 Score :", round(f1_score(y_test, log_pred) * 100, 2), "%")

# -----------------------------
# Random Forest
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n========== Random Forest ==========")
print("Accuracy :", round(accuracy_score(y_test, rf_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, rf_pred) * 100, 2), "%")
print("Recall   :", round(recall_score(y_test, rf_pred) * 100, 2), "%")
print("F1 Score :", round(f1_score(y_test, rf_pred) * 100, 2), "%")

# -----------------------------
# Save Models
# -----------------------------
joblib.dump(log_model, "models/logistic_model.pkl")
joblib.dump(rf_model, "models/random_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModels Saved Successfully!")
print("Saved:")
print("✔ logistic_model.pkl")
print("✔ random_forest.pkl")
print("✔ scaler.pkl")
