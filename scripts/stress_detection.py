#!/usr/bin/env python3
"""
Stress Detection Workflow using WESAD dataset.
This script loads multimodal physiological signals, performs preprocessing and feature extraction,
trains a Random Forest model with stratified cross-validation, and saves the trained model.
"""
import os
import numpy as np
import pandas as pd
import joblib
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

# Set random seed for reproducibility
RANDOM_STATE = 42

def load_wesad_data(data_dir):
    """
    Load WESAD dataset signals from the given directory.
    Expects .pkl files for each subject, each containing:
      - data['signal']['wrist']: wrist-worn device signals (EDA, BVP, ACC, TEMP)
      - data['signal']['chest']: chest-worn device signals (RESP)
      - data['label']: array of labels per sample (0=baseline,1=stress,2=amusement)
    Returns a list of (signals_dict, label_array) for each subject.
    """
    import pickle
    subjects = []
    # Iterate over subject folders (S2, S3, etc.)
    for subj in sorted(os.listdir(data_dir)):
        subj_dir = os.path.join(data_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        fname = f"{subj}.pkl"
        path = os.path.join(subj_dir, fname)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                try:
                    data = pickle.load(f, encoding='latin1')
                except TypeError:
                    data = pickle.load(f)
            # Initialize dict for this subject's signals
            signals = {}
            # Wrist signals (if available)
            if 'signal' in data and 'wrist' in data['signal']:
                wrist = data['signal']['wrist']
                if 'EDA' in wrist:
                    signals['EDA'] = np.array(wrist['EDA']).flatten()
                if 'BVP' in wrist:
                    signals['BVP'] = np.array(wrist['BVP']).flatten()
                if 'TEMP' in wrist:
                    signals['Temp'] = np.array(wrist['TEMP']).flatten()
                if 'ACC' in wrist:
                    # ACC is usually Nx3
                    acc = np.array(wrist['ACC'])
                    signals['ACC'] = acc
            # Chest signals (RESP)
            if 'signal' in data and 'chest' in data['signal']:
                chest = data['signal']['chest']
                if 'RESP' in chest:
                    signals['Resp'] = np.array(chest['RESP']).flatten()
            # Labels array
            if 'label' in data:
                labels = np.array(data['label']).flatten()
                subjects.append((signals, labels))
    return subjects

def extract_features(signals, fs):
    """
    Compute features for given signals in a window.
    signals: dict with keys 'EDA','BVP','Resp','ACC','Temp' (if present)
    fs: sampling frequencies dict for these signals.
    Returns a dict of computed features (means, stds, peaks, etc.).
    """
    features = {}
    # EDA features
    if 'EDA' in signals:
        eda = signals['EDA']
        features['eda_mean'] = np.mean(eda)
        features['eda_std'] = np.std(eda)
        features['eda_max'] = np.max(eda)
        features['eda_min'] = np.min(eda)
        features['eda_skew'] = skew(eda)
        features['eda_kurtosis'] = kurtosis(eda)
        # Phasic peak analysis
        peaks, _ = find_peaks(eda, distance=int(fs['EDA'] * 0.5))
        features['eda_peaks_count'] = len(peaks)
        if len(peaks) > 0:
            features['eda_peaks_mean'] = np.mean(eda[peaks])
            features['eda_peaks_std'] = np.std(eda[peaks])
    # BVP (Blood Volume Pulse) features
    if 'BVP' in signals:
        bvp = signals['BVP']
        features['bvp_mean'] = np.mean(bvp)
        features['bvp_std'] = np.std(bvp)
        # Heart rate analysis from BVP peaks
        peaks, _ = find_peaks(bvp, distance=int(fs['BVP'] * 0.5))
        features['bvp_beats_count'] = len(peaks)
        if len(peaks) > 1:
            # R-R intervals (seconds)
            rr_intervals = np.diff(peaks) / fs['BVP']
            if len(rr_intervals) > 0:
                hr = 60.0 / rr_intervals  # beats per minute
                features['hr_mean'] = np.mean(hr)
                features['hr_std'] = np.std(hr)
    # Respiration features
    if 'Resp' in signals:
        resp = signals['Resp']
        features['resp_mean'] = np.mean(resp)
        features['resp_std'] = np.std(resp)
        # Breathing rate (peaks per minute)
        peaks, _ = find_peaks(resp, distance=int(fs['Resp'] * 1.5))
        breath_count = len(peaks)
        duration_min = len(resp) / fs['Resp'] / 60.0
        features['resp_rate'] = breath_count / duration_min if duration_min > 0 else 0
    # Accelerometer features
    if 'ACC' in signals:
        acc = signals['ACC']
        # Combine 3-axis into magnitude
        if acc.ndim == 1:  # if data is one-dimensional for some reason
            acc_mag = np.abs(acc)
        else:
            acc_mag = np.linalg.norm(acc, axis=1)
        features['acc_mean'] = np.mean(acc_mag)
        features['acc_std'] = np.std(acc_mag)
        features['acc_max'] = np.max(acc_mag)
        features['acc_min'] = np.min(acc_mag)
        peaks, _ = find_peaks(acc_mag, distance=int(fs['ACC'] * 0.1))
        features['acc_peaks_count'] = len(peaks)
    # Temperature features
    if 'Temp' in signals:
        temp = signals['Temp']
        features['temp_mean'] = np.mean(temp)
        features['temp_std'] = np.std(temp)
        features['temp_max'] = np.max(temp)
        features['temp_min'] = np.min(temp)
        if len(temp) > 1:
            features['temp_trend'] = (temp[-1] - temp[0]) / len(temp)
    return features

def segment_signals(signals, labels, window_size, fs):
    """
    Segment signals into non-overlapping windows of length window_size (seconds).
    Returns DataFrame of feature vectors and corresponding label array (mode of labels in window).
    """
    # Determine number of samples per window for each signal type
    window_samples = {sig: int(window_size * fs[sig]) for sig in fs if sig in signals}
    # Use the minimum length across signals for consistent windowing
    n_samples = min(len(sig) for sig in signals.values())
    if n_samples == 0:
        return None, None
    # Slide over windows
    features_list = []
    labels_list = []
    start = 0
    min_win = min(window_samples.values())
    while start + min_win <= n_samples:
        window_data = {sig: (signals[sig][start:start + window_samples[sig]] 
                             if signals[sig].ndim == 1 else signals[sig][start:start + window_samples[sig], :]) 
                       for sig in window_samples}
        # Majority label in this window
        lab_win = labels[start:start + min_win]
        if len(lab_win) == 0:
            break
        # Mode of labels
        unique, counts = np.unique(lab_win, return_counts=True)
        lab = unique[np.argmax(counts)]
        # Extract features for this window
        feats = extract_features(window_data, fs)
        features_list.append(feats)
        labels_list.append(lab)
        start += min_win
    if not features_list:
        return None, None
    return pd.DataFrame(features_list), np.array(labels_list)

def main():
    # Define directories relative to current working directory
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'WESAD')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    # Sampling frequencies for signals (Hz)
    fs = {'EDA': 4, 'BVP': 64, 'Resp': 700, 'ACC': 32, 'Temp': 4}

    print("Loading and preprocessing data...")
    subjects = load_wesad_data(data_dir)
    if not subjects:
        print("No WESAD data files found. Please check the 'WESAD/' directory.")
        return

    # Window size for feature extraction (e.g., 60 seconds)
    window_size = 60
    all_features = []
    all_labels = []
    for signals, labels in subjects:
        feats_df, labs = segment_signals(signals, labels, window_size, fs)
        if feats_df is not None:
            all_features.append(feats_df)
            all_labels.append(labs)
    if not all_features:
        print("No features extracted. Exiting.")
        return
    # Combine data from all subjects
    X = pd.concat(all_features, ignore_index=True).fillna(0.0)
    y = np.concatenate(all_labels)
    # Convert to binary classification: stress (1) vs non-stress (0 or 2 merged)
    y_binary = (y == 1).astype(int)

    # Feature matrix and labels
    X_matrix = X.values
    y_labels = y_binary

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_matrix)

    # Define Random Forest classifier with balanced class weights
    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                 random_state=RANDOM_STATE)

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.zeros_like(y_labels)
    print("Performing StratifiedKFold cross-validation...")
    for train_idx, test_idx in skf.split(X_scaled, y_labels):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train = y_labels[train_idx]
        # Train model
        clf.fit(X_train, y_train)
        # Predict on validation fold
        y_pred[test_idx] = clf.predict(X_test)

    # Print evaluation metrics
    print("Cross-Validation Classification Report:")
    print(classification_report(y_labels, y_pred, target_names=['non-stress','stress']))

    # Feature importance (from last fold model as baseline)
    importances = clf.feature_importances_
    feat_names = X.columns
    # Save feature importance plot
    try:
        import matplotlib.pyplot as plt
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(8,6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feat_names[i] for i in indices], rotation=90)
        plt.ylabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, 'feature_importance.png'))
        plt.close()
        print(f"Feature importance plot saved to '{models_dir}/feature_importance.png'")
    except ImportError:
        print("matplotlib not installed; skipping feature importance plot.")

    # Retrain on full data and save the final model
    print("Training final model on full dataset...")
    clf_final = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                       random_state=RANDOM_STATE)
    clf_final.fit(X_scaled, y_labels)

    # Save model and scaler
    model_path = os.path.join(models_dir, 'stress_model.joblib')
    joblib.dump({'model': clf_final, 'scaler': scaler, 'features': list(feat_names)}, model_path)
    print(f"Trained model saved as '{model_path}'")

if __name__ == "__main__":
    main()
