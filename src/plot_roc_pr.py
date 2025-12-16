import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def main():
    # Load the trained XGBoost model (joblib file)
    model_path = "models/stress_binary_model.joblib"
    if not os.path.exists(model_path):
        # Fallback to root if run from src
        if os.path.exists("../models/stress_binary_model.joblib"):
             model_path = "../models/stress_binary_model.joblib"
        else:
             raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]

    # Load the labeled test dataset
    data_path = "labeled_test_data.csv"
    if not os.path.exists(data_path):
        if os.path.exists("../labeled_test_data.csv"):
            data_path = "../labeled_test_data.csv"
        else:
            raise FileNotFoundError(f"Data not found at {data_path}")

    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)

    # 1. Map labels to binary
    # 2 = Stress -> 1
    # 1, 3, 4 -> 0
    # 0 -> Drop (Transient)
    valid_classes = [1, 2, 3, 4]
    data = data[data["label"].isin(valid_classes)].copy()
    
    target_col = "binary_label"
    data[target_col] = (data["label"] == 2).astype(int)
    
    print(f"Data shape after filtering: {data.shape}")
    print("Label distribution:")
    print(data[target_col].value_counts())

    # Separate features and target
    y_test = data[target_col]
    
    # Ensure features exist
    missing = [f for f in features if f not in data.columns]
    if missing:
        print(f"Warning: Missing features: {missing}")
        # fill missing with 0
        for f in missing:
            data[f] = 0.0
            
    X_test = data[features].fillna(0.0)

    # Predict probabilities for the positive class
    # The pipeline includes 'clf' which is likely RandomForest in user's latest train script, 
    # but the task says "XGBoost". The saved file is 'stress_binary_model.joblib'.
    # We will treat it generically.
    
    print("Predicting probabilities...")
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        try:
            y_scores = model.decision_function(X_test)
        except AttributeError:
            y_scores = model.predict(X_test)
            print("Warning: model has no predict_proba or decision_function, using predict output (0/1).")

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    # Compute Precision-Recall curve and average precision score
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'Model (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    # plt.show() # Blocking, so commented out for automation

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', label=f'Model (AP = {avg_precision:.2f})')
    # Plot no-skill line (the proportion of positive instances)
    no_skill = y_test.mean()
    plt.plot([0, 1], [no_skill, no_skill], color='navy', linestyle='--', label='No Skill')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png')
    # plt.show()

    print("ROC curve plot saved as 'roc_curve.png'")
    print("Precision-Recall curve plot saved as 'precision_recall_curve.png'")

if __name__ == "__main__":
    main()
