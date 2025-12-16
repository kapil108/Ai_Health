import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline as SkPipeline

# Best Parameters from Optimization
# {'clf__max_depth': 3, 'clf__max_features': 'sqrt', 'clf__min_samples_leaf': 5, 'clf__n_estimators': 100}

PARAMS = {
    'n_estimators': 100,
    'max_depth': 3,
    'max_features': 'sqrt',
    'min_samples_leaf': 5,
    'random_state': 42
}

CSV = "wesad_window_features.csv"
OUT_MODEL = "models/stress_model_final.joblib"

def main():
    if not os.path.exists(CSV):
        print(f"Error: {CSV} not found")
        return

    print("Loading data...")
    df = pd.read_csv(CSV)
    
    # Filter and Map
    valid_classes = [1, 2, 3, 4]
    df = df[df["label"].isin(valid_classes)].copy()
    y = (df["label"] == 2).astype(int)
    
    # Prepare X
    if "subject" in df.columns:
        X = df.drop(columns=["label", "subject"])
    else:
        X = df.drop(columns=["label"])
        
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median())
    
    print(f"Training Final Model with params: {PARAMS}")
    print(f"Data Shape: {X.shape}")

    # Create Pipeline (Scaler + RF)
    # Note: SMOTE is ONLY used during training/cross-validation splitting inside the pipeline if using ImbPipeline.
    # For the final production model, we train on the entire dataset. 
    # Whether to use SMOTE on the full dataset before training is a design choice.
    # Usually, we want the model to see the balanced distribution.
    
    # Let's resample first to be explicit, or use class_weight="balanced" as a fallback.
    # Since we optimized with SMOTE pipeline, we should probably stick to that logic or
    # use class_weight="balanced" if we want a simpler inference pipeline.
    # However, to match the optimization exactly, let's use the exact estimator logic.
    
    # Pipeline for training on full data:
    # 1. Scale
    # 2. Over-sample (SMOTE)
    # 3. Train RF
    
    # Pipeline for INFERENCE (saved model):
    # 1. Scale
    # 2. Predict (RF)
    # SMOTE is not needed at inference time.
    
    # So we do:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    clf = RandomForestClassifier(**PARAMS)
    clf.fit(X_resampled, y_resampled)
    
    # Create the inference pipeline
    final_pipeline = SkPipeline([
        ('scaler', scaler), # This scaler is already fitted? No, we need to pass the fitted scaler or re-create pipeline logic.
        # SkPipeline doesn't accept already fitted objects directly in the list definition usually, 
        # but we can set the steps.
        # Easier way: Define pipeline, fit it. But SMOTE breaks SkPipeline (it's only in ImbPipeline).
        # We need a plain SkPipeline for inference.
        
        ('clf', clf)
    ])
    
    # Re-construct pipeline properly
    # We need to save the scaler that was fitted on X
    # And the CLF that was fitted on X_resampled
    
    # Manually constructing the object to save
    model_bundle = {
        "pipeline": SkPipeline([
            ('scaler', scaler),
            ('clf', clf)
        ]),
        "features": list(X.columns)
    }
    
    joblib.dump(model_bundle, OUT_MODEL)
    print(f"Saved optimized model to {OUT_MODEL}")
    
    # Validate quick stats
    acc = clf.score(X_scaled, y) # This is on original unbalanced data
    print(f"Training Accuracy (Unbalanced Check): {acc:.4f}")

if __name__ == "__main__":
    main()
