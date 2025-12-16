import pandas as pd
import joblib
import numpy as np
import os

# Load saved model and scaler
model_path = "models/stress_binary_model.joblib"
if not os.path.exists(model_path):
    # Try alternate path if run from src
    if os.path.exists("../models/stress_binary_model.joblib"):
        model_path = "../models/stress_binary_model.joblib"
    else:
        raise FileNotFoundError(f"Could not find {model_path}. Please run training first.")

print(f"Loading model from {model_path}...")
model_bundle = joblib.load(model_path)
model = model_bundle["model"]
features = model_bundle["features"]

# Load test data
input_csv = "test_sample.csv"
if not os.path.exists(input_csv):
    # Try alternate
    if os.path.exists("../test_sample.csv"):
        input_csv = "../test_sample.csv"
    else:
        # Fallback for demo if users haven't created it yet
        print("Warning: test_sample.csv not found. Please provide input data.")
        exit(1)

print(f"Loading test data from {input_csv}...")
test_data = pd.read_csv(input_csv)

# Ensure all required features are present
# We filter ONLY the features used by the model
missing_features = [f for f in features if f not in test_data.columns]
if missing_features:
    print(f"Warning: Missing {len(missing_features)} features. Filling with 0.0.")
    for f in missing_features:
        test_data[f] = 0.0

# Select and order columns to match model
X_test = test_data[features].fillna(0.0)

# Predict
# The model is a Pipeline that includes scaling, so we pass X_test directly
print("Running prediction...")
predictions = model.predict(X_test)
pred_labels = ["Non-Stress" if p == 0 else "Stress" for p in predictions]

# Save or print predictions
output_csv = "predicted_output.csv"
test_data["Prediction"] = pred_labels
test_data.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")
print(test_data[["Prediction"]].head())
