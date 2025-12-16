import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

CSV = "wesad_window_features.csv"
RANDOM_STATE = 42

def main():
    if not os.path.exists(CSV):
        print(f"Error: {CSV} not found")
        return

    print("Loading data...")
    df = pd.read_csv(CSV)
    
    # Filter and Map (Same as before)
    valid_classes = [1, 2, 3, 4]
    df = df[df["label"].isin(valid_classes)].copy()
    y = (df["label"] == 2).astype(int)
    
    X = df.drop(columns=["label", "subject"])
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median())
    
    print(f"Data Shape: {X.shape}")
    print(f"Class Balance:\n{y.value_counts()}")

    # Define Pipeline
    pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
    ])

    # Define Parameter Grid
    # We want to LIMIT complexity
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 5, 8, None], # Crucial: prevent infinite depth
        'clf__max_features': ['sqrt', 'log2'],
        'clf__min_samples_leaf': [2, 5, 10] # Require more samples to make a decision
    }

    print("\nStarting Grid Search (this may take a minute)...")
    
    # Use F1-score for Stress (binary class 1) as the optimization metric
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5, # 5-Fold Stratified
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X, y)
    
    print("\noptimization Complete!")
    print(f"Best F1 Score: {grid.best_score_:.4f}")
    print("Best Parameters:")
    for k, v in grid.best_params_.items():
        print(f"  {k}: {v}")
        
    # Save best params to a file for the next script to read (or manual inspection)
    with open("best_params.txt", "w") as f:
        f.write(str(grid.best_params_))

if __name__ == "__main__":
    main()
