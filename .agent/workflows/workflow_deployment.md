---
description: How to deploy the Stress Detection App to Render and Streamlit Cloud
---
# How to Deploy Your AI Health Project

This guide explains how to host your project for free using **Render** (Backend) and **Streamlit Community Cloud** (Frontend).

## Prerequisites
1.  **GitHub Account**: Your project code must be pushed to a GitHub repository.
2.  **Gemini API Key**: You need your API key handy.

---

## Part 1: Deploy Backend (FastAPI) on Render

1.  **Sign Up/Login**: Go to [render.com](https://render.com) and log in with GitHub.
2.  **New Web Service**: Click "New +" -> "Web Service".
3.  **Connect Repo**: Select your `my_ai_health_project` repository.
4.  **Configure Settings**:
    *   **Name**: `stress-api` (or similar)
    *   **Runtime**: `Python 3`
    *   **Build Command**: `pip install -r requirements.txt`
    *   **Start Command**: `uvicorn src.api:app --host 0.0.0.0 --port 10000`
5.  **Environment Variables** (Advanced):
    *   Click "Add Environment Variable".
    *   Key: `GEMINI_API_KEY`
    *   Value: `your-actual-api-key-here`
6.  **Deploy**: Click "Create Web Service".
7.  **Wait**: It will take a few minutes. Once live, copy the URL (e.g., `https://stress-api.onrender.com`).

---

## Part 2: Deploy Frontend (Dashboard) on Streamlit Cloud

1.  **Sign Up/Login**: Go to [streamlit.io/cloud](https://streamlit.io/cloud) and log in with GitHub.
2.  **New App**: Click "New app".
3.  **Select Repo**: Choose `my_ai_health_project`.
4.  **Main File Path**: Enter `src/dashboard.py`.
5.  **Advanced Settings** (Crucial Step):
    *   Click "Advanced settings".
    *   **Environment Variables**:
        *   Key: `API_URL`
        *   Value: `https://stress-api.onrender.com` (The URL from Part 1, **without** a trailing slash).
        *   *(Optional)* Key: `GEMINI_API_KEY` (Only if you want LLM features specifically in the dashboard code, though the backend handles most of it).
6.  **Deploy**: Click "Deploy".

---

## Part 3: Verify

1.  Open your new Streamlit App URL.
2.  Upload `test_sample.csv` or try "Manual Data Entry".
3.  Calculations run on Render, and results appear on Streamlit!

> [!TIP]
> **Free Tier Delay**: Render's free tier "spins down" after inactivity. The first request might take 50 seconds. This is normal.
