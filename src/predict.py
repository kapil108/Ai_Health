import pandas as pd
import joblib
import numpy as np
import os

# Load saved model and scaler
# Adjust path if script is run from src/ or root. Assuming run from root for now based on user snippet.
# If run from src, it should be "../models/..."
# But user snippet was: "models/xgb_stress_model.joblib"
# I'll stick to user snippet but maybe add a check or comment.
try:
    model_bundle = joblib.load("models/xgb_stress_model.joblib")
except FileNotFoundError:
    # Try alternate path if run from src
    try:
        model_bundle = joblib.load("../models/xgb_stress_model.joblib")
    except FileNotFoundError:
        raise FileNotFoundError("Could not find models/xgb_stress_model.joblib. Please run from project root or check path.")

model = model_bundle["model"]
scaler = model_bundle["scaler"]
features = model_bundle["features"]

# Load test data (replace with your CSV)
input_csv = "test_sample.csv"
if not os.path.exists(input_csv):
    # Just for robustness if user didn't provide one, we might want to warn or exit, 
    # but for this script I'll assume user provides it or I'll handle it below.
    pass

try:
    test_data = pd.read_csv(input_csv)
except FileNotFoundError:
    print(f"Error: {input_csv} not found. Please provide a valid CSV file.")
    exit(1)

# Ensure all required features are present
missing_features = [f for f in features if f not in test_data.columns]
if missing_features:
    raise ValueError(f"Missing required features in test data: {missing_features}")

# Reorder and scale features
X_test = test_data[features].fillna(0.0)
X_scaled = scaler.transform(X_test)

# Predict
predictions = model.predict(X_scaled)
pred_labels = ["Non-Stress" if p == 0 else "Stress" for p in predictions]

# Save or print predictions
test_data["Prediction"] = pred_labels
test_data.to_csv("predicted_output.csv", index=False)
print("Predictions saved to predicted_output.csv")
