
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import joblib
import os
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8001/predict"
MANUAL_API_URL = "http://127.0.0.1:8001/predict_manual"

st.set_page_config(page_title="AI Stress Detector", layout="wide")

st.title("🩺 AI Stress Detector (Demo)")
st.markdown("Upload a CSV or enter values manually to detect stress.")

# Tabs for separate functionality
tab1, tab2 = st.tabs(["📂 Batch File Upload", "✍️ Manual Input"])

# --- TAB 1: Batch Upload (Existing Logic) ---
with tab1:
    st.markdown("### Upload a CSV of features")
    with st.expander("Download sample CSV / feature template"):
        st.write("Sample columns expected (example): `bvp_mean,bvp_std,eda_mean,eda_std,hr_mean,acc_mean`")
        if st.button("Download sample CSV"):
            sample = pd.DataFrame([{
                "bvp_mean": 0.38, "bvp_std": 26.9, "eda_mean": 0.45, "eda_std": 0.12, "hr_mean": 76.4, "acc_mean": 0.03
            }])
            st.download_button("Download sample", data=sample.to_csv(index=False), file_name="sample_test.csv")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="batch_upload")
    
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.subheader("Preview uploaded data")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Send to API and Predict"):
                with st.spinner("Sending to API..."):
                    try:
                        # send file bytes to API
                        uploaded.seek(0)
                        files = {"file": ("upload.csv", uploaded.read(), "text/csv")}
                        resp = requests.post(API_URL, files=files, timeout=30)
                        resp.raise_for_status()
                        result = resp.json()
                        preds = result.get("predictions")
                        probs = result.get("probabilities") or None
                        
                        # compose results dataframe
                        out = df.copy()
                        out["predicted_label"] = ["Stress" if int(p)==1 else "Non-Stress" for p in preds]
                        if probs is not None:
                            out["probability"] = probs
                        else:
                            out["probability"] = None
                        
                        # risk level
                        def risk_from_prob(p):
                            if p is None: return "Unknown"
                            p = float(p)
                            if p >= 0.75: return "High"
                            if p >= 0.5: return "Medium"
                            return "Low"
                        out["risk"] = out["probability"].apply(risk_from_prob)
                        
                        st.success("Prediction complete.")
                        st.subheader("Summary")
                        total = len(out)
                        n_stress = (out["predicted_label"]=="Stress").sum()
                        avg_prob = out["probability"].dropna().mean() if out["probability"].notna().any() else None
                        cols = st.columns(3)
                        cols[0].metric("Rows processed", total)
                        cols[1].metric("Predicted Stress", f"{n_stress} ({n_stress/total:.0%})")
                        cols[2].metric("Avg Probability", f"{avg_prob:.2f}" if avg_prob is not None else "N/A")
                        
                        st.subheader("Predictions")
                        st.dataframe(out)

                        # bar chart
                        st.bar_chart(out["predicted_label"].value_counts())
                        
                    except Exception as e:
                         st.error(f"API request failed: {e}")

        with col2:
            st.info("Model info & tips")
            st.write("- Model: Binary Stress Detector")
            st.markdown("**If stress predicted**: take a break.")
            if st.checkbox("Show ROC & PR curves"):
                if os.path.exists("roc_curve.png"):
                    st.image("roc_curve.png", caption="ROC Curve")
                if os.path.exists("precision_recall_curve.png"):
                    st.image("precision_recall_curve.png", caption="Precision-Recall Curve")

# --- TAB 2: Manual Input (New Logic) ---
with tab2:
    st.markdown("### Enter specific feature values")
    with st.form("manual_input_form"):
        col_A, col_B = st.columns(2)
        with col_A:
            bvp_mean = st.number_input("BVP Mean", value=0.0)
            bvp_std  = st.number_input("BVP Std", value=0.0)
            eda_mean = st.number_input("EDA Mean", value=0.0)
            eda_std  = st.number_input("EDA Std", value=0.0)
        with col_B:
            eda_max  = st.number_input("EDA Max", value=0.0)
            hr_mean  = st.number_input("HR Mean (optional)", value=0.0)
            acc_mean = st.number_input("ACC Mean (optional)", value=0.0)
        
        submitted = st.form_submit_button("Predict Feature Set")
        
        if submitted:
            payload = {
                "bvp_mean": bvp_mean,
                "bvp_std": bvp_std,
                "eda_mean": eda_mean,
                "eda_std": eda_std,
                "eda_max": eda_max,
                "hr_mean": hr_mean,
                "acc_mean": acc_mean
            }
            try:
                resp = requests.post(MANUAL_API_URL, json=payload, timeout=10)
                resp.raise_for_status()
                j = resp.json()
                label = j.get("prediction")
                conf = j.get("confidence")
                
                if label == "Stress":
                    st.error(f"🧠 Prediction: **{label}** (Confidence: {conf}%)")
                    st.markdown("⚠️ High stress detected. Consider taking a deep breath.")
                else:
                    st.success(f"🧠 Prediction: **{label}** (Confidence: {conf}%)")
                    st.markdown("✅ You seem to be in a non-stress state.")
                    
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
