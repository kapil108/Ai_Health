import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def main():
    # 1. Load Model
    model_path = "models/stress_binary_model.joblib"
    if not os.path.exists(model_path):
        print("Model not found!")
        return
        
    print(f"Loading model from {model_path}...")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    
    # 2. Load Data
    data_path = "wesad_window_features.csv"
    if not os.path.exists(data_path):
        print("Data file not found!")
        return
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Filter valid classes like in training
    valid_classes = [1, 2, 3, 4]
    df = df[df["label"].isin(valid_classes)].copy()
    
    # Map to Binary (Stress=2 -> 1, Others -> 0)
    df["binary_label"] = (df["label"] == 2).astype(int)
    
    # Prepare X
    X = df.drop(columns=["label", "binary_label", "subject"])
    # Ensure numeric
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median()) # Simple fill for analysis
    
    y_true = df["binary_label"]
    
    # 3. Get Probabilities
    if hasattr(model, "predict_proba"):
        print("Predicting probabilities...")
        probs = model.predict_proba(X)[:, 1]
    else:
        print("Model does not support predict_proba!")
        return

    # 4. Analyze Thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("\n--- Threshold Analysis (Full Dataset) ---")
    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Accuracy':<10}")
    print("-" * 55)
    
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        print(f"{t:<10.1f} {prec:<10.3f} {rec:<10.3f} {f1:<10.3f} {acc:<10.3f}")

if __name__ == "__main__":
    main()
