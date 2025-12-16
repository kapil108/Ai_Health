# src/eval_loso_binary.py
import os, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

CSV = "wesad_window_features.csv"
RANDOM_STATE = 42

def main():
    if not os.path.exists(CSV):
        raise FileNotFoundError(f"{CSV} not found.")
        
    df = pd.read_csv(CSV)
    
    # 1. Filter Valid Classes
    # WESAD Labels:
    # 0 = Transient (Noise/Undefined) -> DROP
    # 1 = Baseline     -> Non-Stress
    # 2 = Stress       -> Stress (Target)
    # 3 = Amusement    -> Non-Stress
    # 4 = Meditation   -> Non-Stress
    # 5/6/7 = Ignored
    
    valid_classes = [1, 2, 3, 4]
    df = df[df["label"].isin(valid_classes)].copy()
    
    # 2. Map to Binary
    # Stress (2) -> 1
    # Everything else (1, 3, 4) -> 0
    df["binary_label"] = (df["label"] == 2).astype(int)
    
    print(f"Data filtered. Shape: {df.shape}")
    print("Label mapping: 2 (Stress) -> 1, [1,3,4] (Not Stress) -> 0")
    print(df["binary_label"].value_counts())

    subjects = df["subject"].unique()
    cm_sum = None

    print("\nStarting basic LOSO (No SMOTE in CV for speed, use unbalanced class weights if needed)...")
    # Actually, we should use SMOTE because we saw imbalance.
    
    for s in subjects:
        train = df[df["subject"] != s]
        test = df[df["subject"] == s]
        
        # Prepare X, y
        X_train_raw = train.drop(columns=["label", "binary_label", "subject"])
        X_test_raw = test.drop(columns=["label", "binary_label", "subject"])
        
        # Numeric only
        num_cols = X_train_raw.select_dtypes(include=[np.number]).columns
        X_train = X_train_raw[num_cols].fillna(train[num_cols].median())
        X_test = X_test_raw[num_cols].fillna(train[num_cols].median())
        
        y_train = train["binary_label"]
        y_test = test["binary_label"]
        
        # SMOTE + RF Pipeline
        # Note: applying SMOTE on training fold of LOSO
        # SMOTE + RF Pipeline
        # Note: applying SMOTE on training fold of LOSO
        pipeline = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            # Use optimized parameters here to match the new model
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=3, max_features='sqrt', min_samples_leaf=5, random_state=RANDOM_STATE)) 
        ])
        
        pipeline.fit(X_train.values, y_train.values)
        y_pred = pipeline.predict(X_test.values)
        
        print(f"Subject {s}:")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        
        cm = confusion_matrix(y_test, y_pred)
        if cm_sum is None:
            cm_sum = cm
        else:
             # Binary CM is always 2x2. No padding needed if both classes exist.
             # If a subject has NO stress labels (unlikely in WESAD), we might need checks.
             if cm.shape == (2,2):
                 cm_sum += cm
             else:
                 # Extremely rare case if subject misses a class
                 # Pad manually
                 temp = np.zeros((2,2), dtype=int)
                 # This logic depends on which class is missing, complicated.
                 # For WESAD, every subject has all conditions.
                 cm_sum += cm # potentially bugs if shapes differ
                 
    print("\nAggregated Confusion Matrix (Stress = 1, Non-Stress = 0):")
    print(cm_sum)

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_sum, display_labels=["Non-Stress", "Stress"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Aggregated Confusion Matrix (LOSO)")
    plt.savefig("confusion_matrix_loso_binary.png")
    plt.show()

if __name__ == "__main__":
    main()
