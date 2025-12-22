import streamlit as st
import pandas as pd
import requests
import io

import os

# Configuration
# Default to localhost for dev, but allow Render/Streamlit Cloud to override
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Stress Detection Dashboard", layout="wide")

st.title("🧠 AI Stress Detection Dashboard")

# Sidebar for Navigation
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choose Input Method:", ["Upload CSV File", "Manual Data Entry"])

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Instructions
1. **Upload CSV File**: Upload a dataset containing physiological features to get batch predictions and visualize data.
2. **Manual Data Entry**: Input specific feature values to get a single stress prediction.
""")

if mode == "Upload CSV File":
    st.header("📂 Batch Analysis (CSV)")
    st.markdown("Upload your physiological data (CSV) to detect stress levels.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        st.info("File uploaded successfully.")
        
        # Read Data
        df = pd.read_csv(uploaded_file)
        
        # 1. Show Data Preview
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())
        
        # 2. Visualize Data
        st.subheader("📈 Data Visualizations")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("Select feature to visualize:", numeric_cols)
            st.line_chart(df[selected_col])
        else:
            st.write("No numeric data to visualize.")

        # 3. Analyze Button
        st.markdown("---")
        if st.button("Analyze Stress Levels"):
            # Reset Chat History for new analysis
            st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm here to help you understand your stress levels. Ask me anything about your results or stress management."}]
            
            with st.spinner("Analyzing data with AI model..."):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                    response = requests.post(f"{API_URL}/predict", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Handle predictions
                        preds = data.get("predictions", [])
                        results = pd.DataFrame(preds, columns=["prediction"])
                        
                        # Map numerical predictions
                        results["prediction"] = results["prediction"].apply(lambda x: "Stress" if x == 1 else "Non-Stress")

                        st.success("Analysis Complete!")
                        
                        # Display AI Advice if available
                        summary_advice = data.get("summary_advice")
                        if summary_advice:
                            st.info(f"🤖 **AI Health Coach Insights:**\n\n{summary_advice}")
                        
                        # Store context for Chatbot
                        st.session_state['current_context'] = (
                            f"Use this context for the user's current batch analysis:\n"
                            f"- Total Samples: {total_samples}\n"
                            f"- Stress Detected: {stress_count} ({stress_percentage:.1f}%)\n"
                            f"- Summary Advice: {summary_advice if summary_advice else 'None'}\n"
                        )
                        
                        # Metrics
                        total_samples = len(results)
                        stress_count = results[results["prediction"] == "Stress"].shape[0]
                        stress_percentage = (stress_count / total_samples) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Samples", total_samples)
                        col2.metric("Stress Detected", stress_count)
                        col3.metric("Stress %", f"{stress_percentage:.1f}%")
                        
                        # Prediction Distribution
                        st.subheader("Prediction Distribution")
                        st.bar_chart(results["prediction"].value_counts())
                        
                        # Detailed Results
                        st.subheader("Detailed Results")
                        st.dataframe(results)
                        
                        # Download key
                        csv = results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Predictions",
                            csv,
                            "stress_predictions.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error(f"Could not connect to the API at {API_URL}. Is the backend running?")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

elif mode == "Manual Data Entry":
    st.header("✍️ Manual Prediction")
    st.markdown("Enter physiological feature values to predict stress.")
    
    with st.form("manual_input_form"):
        st.markdown("### Physiological Indicators")
        col1, col2 = st.columns(2)
        
        with col1:
            bvp_mean = st.number_input("BVP Mean", value=0.0, format="%.4f", help="Blood Volume Pulse (Mean): Measure of blood flow volume.")
            bvp_std = st.number_input("BVP Std Dev", value=0.0, format="%.4f", help="Blood Volume Pulse (Standard Deviation): Variability in blood flow.")
            bvp_max = st.number_input("BVP Max", value=0.0, format="%.4f", help="Blood Volume Pulse (Maximum): Peak blood flow volume.")
            eda_mean = st.number_input("EDA Mean", value=0.0, format="%.4f", help="Electrodermal Activity (Mean): Skin conductance, related to sweat and physiological arousal.")
        
        with col2:
            eda_std = st.number_input("EDA Std Dev", value=0.0, format="%.4f", help="Electrodermal Activity (Standard Deviation): Variability in skin conductance.")
            eda_max = st.number_input("EDA Max", value=0.0, format="%.4f", help="Electrodermal Activity (Maximum): Peak skin conductance.")
            hr_mean = st.number_input("Heart Rate Mean (Optional - Unused)", value=0.0, format="%.4f", help="Average Heart Rate (beats per minute). Currently unused by the model.")
            acc_mean = st.number_input("Accelerometer Mean (Optional - Unused)", value=0.0, format="%.4f", help="Average motion/movement intensity. Currently unused by the model.")
            
        submitted = st.form_submit_button("Predict Stress")
        
        if submitted:
            # Reset Chat History for new prediction
            st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm here to help you understand your stress levels. Ask me anything about your results or stress management."}]
            
            st.session_state['manual_success'] = True
            payload = {
                "bvp_mean": bvp_mean,
                "bvp_std": bvp_std,
                "bvp_max": bvp_max,
                "eda_mean": eda_mean,
                "eda_std": eda_std,
                "eda_max": eda_max,
                "hr_mean": hr_mean,
                "acc_mean": acc_mean
            }
            
            with st.spinner("Predicting (reading biosignals)..."):
                try:
                    response = requests.post(f"{API_URL}/predict_manual", json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result.get("prediction", "Unknown")
                        confidence = result.get("confidence", 0.0)
                        advice = result.get("advice")
                        
                        # Store context for Chatbot
                        st.session_state['current_context'] = (
                            f"Use this context for the user's current manual reading:\n"
                            f"- BVP Mean: {bvp_mean}\n"
                            f"- BVP Std Dev: {bvp_std}\n"
                            f"- BVP Max: {bvp_max}\n"
                            f"- EDA Mean: {eda_mean}\n"
                            f"- EDA Std Dev: {eda_std}\n"
                            f"- EDA Max: {eda_max}\n"
                            f"- Heart Rate Mean: {hr_mean}\n"
                            f"- Accelerometer Mean: {acc_mean}\n"
                            f"- Prediction: {prediction} (Confidence: {confidence}%)"
                        )
                        
                        st.write("---")
                        st.subheader("Result")
                        if prediction == "Stress":
                            st.error(f"🚨 **Stress Detected** (Confidence: {confidence}%)")
                            
                            if advice:
                                st.info(f"🤖 **AI Health Coach Advice:**\n\n{advice}")
                            else:
                                st.warning("Advice could not be generated. Please check your backend logs or .env configuration.")

                        else:
                            st.success(f"✅ **Non-Stress** (Confidence: {confidence}%)")
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error(f"Could not connect to the API at {API_URL}. Is the backend running?")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# --- Chat System ---
# Check if CSV is uploaded OR if Manual Entry was successfully submitted
manual_active = (mode == "Manual Data Entry" and st.session_state.get('manual_success', False))
csv_active = (mode == "Upload CSV File" and uploaded_file is not None)

if csv_active or manual_active:
    st.markdown("---")
    st.header("💬 Chat with AI Health Coach")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm here to help you understand your stress levels. Ask me anything about your results or stress management."}]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your stress..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    hist_payload = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                    
                    # Use dynamically stored context if available, else default
                    context_str = st.session_state.get('current_context', "User is on dashboard.")
                    
                    payload = {"message": prompt, "context": context_str, "history": hist_payload}
                    
                    response = requests.post(f"{API_URL}/chat", json=payload)
                    
                    if response.status_code == 200:
                        ans = response.json().get("response", "No response.")
                        st.markdown(ans)
                        st.session_state.messages.append({"role": "assistant", "content": ans})
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
