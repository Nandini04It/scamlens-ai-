"""
utils.py
--------
ScamLens AI - UPI Fraud Analytics
Helper functions: data cleaning, feature engineering, model training,
prediction, and recommendations.

This file mirrors the exact logic used in the ScamLens_AI.ipynb notebook,
so results in the Streamlit app match the notebook.
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ==========================================================
# STEP 1 : DATA CLEANING
# ==========================================================
def load_and_clean_data(file_path_or_buffer):
    """
    Loads the raw scam_dataset.csv and applies the same cleaning
    steps used in the notebook (Step 4 : Data Cleaning).
    """
    df = pd.read_csv(file_path_or_buffer)

    # Convert Date Format
    df["date"] = pd.to_datetime(df["date"])

    # Convert Time Format (keep as datetime.time objects)
    df["time"] = pd.to_datetime(df["time"]).dt.time

    # Convert Amount into Numeric (handles ₹ symbol / commas if present)
    df["amount_inr"] = (
        df["amount_inr"]
        .astype(str)
        .str.replace("[₹,]", "", regex=True)
        .astype(float)
    )

    return df


# ==========================================================
# STEP 2 : FEATURE ENGINEERING
# ==========================================================
def get_time_slot(t):
    hour = t.hour
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


def fraud_severity(amount):
    if amount < 5000:
        return "Low"
    elif amount < 20000:
        return "Medium"
    else:
        return "High"


def engineer_features(df):
    """
    Adds all engineered features used in the notebook
    (Step 5 : Feature Engineering).
    """
    df = df.copy()

    # Feature 1 : Amount Category
    df["Amount_Category"] = pd.cut(
        df["amount_inr"],
        bins=[0, 1000, 10000, float("inf")],
        labels=["Low", "Medium", "High"],
    )

    # Feature 2 : Time Slot
    df["Time_Slot"] = df["time"].apply(get_time_slot)

    # Feature 3 : Weekend / Weekday
    df["Day_Name"] = df["date"].dt.day_name()
    df["Day_Type"] = df["Day_Name"].apply(
        lambda x: "Weekend" if x in ["Saturday", "Sunday"] else "Weekday"
    )

    # Feature 4 : Fraud Severity (target for ML model)
    df["Fraud_Severity"] = df["amount_inr"].apply(fraud_severity)

    return df


def full_pipeline(file_path_or_buffer):
    """Runs cleaning + feature engineering in one call."""
    df = load_and_clean_data(file_path_or_buffer)
    df = engineer_features(df)
    return df


# ==========================================================
# STEP 3 : MODEL TRAINING (Decision Tree)
# ==========================================================
FEATURES = [
    "amount_inr",
    "Time_Slot",
    "Day_Type",
    "upi_app",
    "transaction_type",
    "fraud_lure",
]
TARGET = "Fraud_Severity"
CATEGORICAL_COLUMNS = ["Time_Slot", "Day_Type", "upi_app", "transaction_type", "fraud_lure"]

MODEL_PATH = "fraud_severity_model.pkl"
ENCODERS_PATH = "feature_encoders.pkl"
TARGET_ENCODER_PATH = "target_encoder.pkl"


def train_model(df):
    """
    Trains the Decision Tree Classifier exactly as in the notebook
    (Step 7 : AI Feature) and returns everything the app needs.
    """
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=target_encoder.classes_, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "classes": target_encoder.classes_,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_names": X.columns.tolist(),
    }

    return model, encoders, target_encoder, metrics


def save_model(model, encoders, target_encoder):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(target_encoder, TARGET_ENCODER_PATH)


def load_saved_model():
    """Loads previously saved model + encoders, if present on disk."""
    if all(os.path.exists(p) for p in [MODEL_PATH, ENCODERS_PATH, TARGET_ENCODER_PATH]):
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        target_encoder = joblib.load(TARGET_ENCODER_PATH)
        return model, encoders, target_encoder
    return None, None, None


def get_or_train_model(df):
    """
    Tries to load a saved model first (fast). If not found,
    trains a fresh model on the given dataframe and saves it.
    """
    model, encoders, target_encoder = load_saved_model()
    metrics = None
    if model is None:
        model, encoders, target_encoder, metrics = train_model(df)
        save_model(model, encoders, target_encoder)
    return model, encoders, target_encoder, metrics


# ==========================================================
# STEP 4 : PREDICTION
# ==========================================================
def predict_severity(model, encoders, target_encoder, amount, time_slot,
                      day_type, upi_app, transaction_type, fraud_lure):
    """
    Builds a single-row dataframe from user inputs, encodes it using the
    same encoders used in training, and returns the predicted severity.
    """
    sample = pd.DataFrame({
        "amount_inr": [amount],
        "Time_Slot": [time_slot],
        "Day_Type": [day_type],
        "upi_app": [upi_app],
        "transaction_type": [transaction_type],
        "fraud_lure": [fraud_lure],
    })

    for col in sample.columns:
        if col in encoders:
            sample[col] = encoders[col].transform(sample[col])

    sample = sample[FEATURES]  # ensure correct column order

    prediction = model.predict(sample)[0]
    severity = target_encoder.inverse_transform([prediction])[0]
    return severity


def get_recommendation(severity):
    """Returns risk level, emoji, and a list of recommendation tips."""
    if severity == "High":
        return {
            "level": "HIGH RISK",
            "emoji": "🔴",
            "tips": [
                "❌ Avoid this transaction",
                "🏦 Contact your bank immediately",
                "🚨 Report the scam (cybercrime.gov.in / 1930 helpline)",
                "🔒 Verify the sender/receiver before paying",
            ],
        }
    elif severity == "Medium":
        return {
            "level": "MEDIUM RISK",
            "emoji": "🟡",
            "tips": [
                "⚠ Verify the receiver's identity",
                "📲 Double-check payment details",
                "🔍 Avoid clicking unknown links",
            ],
        }
    else:
        return {
            "level": "LOW RISK",
            "emoji": "🟢",
            "tips": [
                "✅ Transaction looks safe",
                "✔ Still verify the receiver as a good habit",
            ],
        }