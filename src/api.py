import pandas as pd
import joblib
import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import google.api_core.exceptions
from typing import Optional, List, Dict

load_dotenv()

import faiss

# Global variables for model
model_bundle = None
model = None
features = None

# FAISS Vector DB Init
DIMENSION = 768 # Gemini Embedding Dimension
faiss_index = faiss.IndexFlatL2(DIMENSION)
stored_memories = [] # To store actual text corresponding to vectors

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model_bundle, model, features
    model_path = "models/stress_binary_model.joblib"
    if not os.path.exists(model_path):
        # Allow running from src or root
        if os.path.exists("../models/stress_binary_model.joblib"):
            model_path = "../models/stress_binary_model.joblib"
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    features = model_bundle["features"]
    print("Model loaded successfully.")
    
    # Initialize Gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        print("Gemini API configured.")
    else:
        print("Warning: GEMINI_API_KEY not found in environment.")
        
    yield
    # Clean up (if needed)

app = FastAPI(title="Stress Detection API", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity (or specify Streamlit URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Stress Detection API is running. POST CSV to /predict"}

@app.post("/predict")
async def predict_stress(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Validation: Ensure features exist
        missing = [f for f in features if f not in df.columns]
        if missing:
            for f in missing:
                df[f] = 0.0
        
        # Select features in order
        X = df[features].fillna(0.0)
        
        # Predict
        predictions = model.predict(X)
        
        # Try to get probabilities
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[:, 1].tolist()
        elif hasattr(model, "decision_function"):
             pass
        
        # Calculate Summary Advice if Stress Detected
        advice = None
        stress_indices = np.where(predictions == 1)[0]
        
        # Only generate advice if we have enough stress samples and the key is set
        if len(stress_indices) > 0:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    # Calculate aggregates for stress periods
                    stress_data = X.iloc[stress_indices]
                    
                    # specific features for prompt
                    mean_hr = stress_data.get('hr_mean', pd.Series([0])).mean()
                    mean_eda = stress_data.get('eda_mean', pd.Series([0])).mean()
                    mean_bvp = stress_data.get('bvp_mean', pd.Series([0])).mean()
                    
                    genai.configure(api_key=api_key)
                    model_llm = genai.GenerativeModel('models/gemini-flash-latest')
                    
                    prompt = f"""
                    Analyze this batch stress data summary:
                    - Detected Stress Samples: {len(stress_indices)}
                    - Avg Heart Rate during stress: {mean_hr:.1f}
                    - Avg EDA during stress: {mean_eda:.1f}
                    - Avg BVP during stress: {mean_bvp:.1f}
                    
                    Provide a brief (2-3 sentences) summary of the user's overall stress pattern and 2 key recommendations for post-session recovery.
                    """
                    
                    response = model_llm.generate_content(prompt)
                    advice = response.text
                    
                    # Store in Memory
                    memory_text = f"Batch Analysis - Stress Count: {len(stress_indices)}. Stats: HR {mean_hr:.1f}, EDA {mean_eda:.1f}. AI Advice: {advice}"
                    add_to_memory(memory_text)
                    
                except Exception as e:
                    print(f"Batch LLM Error: {e}")
                    advice = "Could not generate batch summary advice."

        # Return structured response
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities,
            "summary_advice": advice
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ManualInput(BaseModel):
    bvp_mean: float
    bvp_std: float
    bvp_max: float
    eda_mean: float
    eda_std: float
    eda_max: float
    hr_mean: float = 0.0
    acc_mean: float = 0.0
    gemini_api_key: Optional[str] = None 

@app.post("/predict_manual")
async def predict_manual(data: ManualInput):
    try:
        # Convert input to DataFrame to match pipeline expectation
        input_dict = data.dict(exclude={"gemini_api_key"})
        
        # Create one-row DataFrame
        df = pd.DataFrame([input_dict])
        
        # Validate/Fill missing features
        missing = [f for f in features if f not in df.columns]
        for f in missing:
            df[f] = 0.0
            
        # Select features in order
        X = df[features].fillna(0.0)
        
        # Predict
        prediction = model.predict(X)[0]
        
        # Probability
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            confidence = probs[0].max() * 100
            
        pred_label = "Stress" if prediction == 1 else "Non-Stress"
        
        advice = None
        # Call Gemini if stress is detected
        if pred_label == "Stress":
            # Use key from request or env
            key_to_use = data.gemini_api_key or os.getenv("GEMINI_API_KEY")
            
            if key_to_use:
                try:
                    genai.configure(api_key=key_to_use)
                    model_llm = genai.GenerativeModel('models/gemini-flash-latest')
                    
                    prompt = f"""
                    You are an empathetic AI health coach. A user has been detected with high stress levels based on their physiological data:
                    - Heart Rate (Mean): {data.hr_mean:.1f}
                    - EDA (Mean): {data.eda_mean:.1f}
                    - BVP (Mean): {data.bvp_mean:.1f}
                    
                    Please provide 3 immediate, short, and actionable tips to help them relieve stress right now. 
                    Be supportive and concise.
                    """
                    
                    response = model_llm.generate_content(prompt)
                    advice = response.text
                    
                    # Store in Memory
                    memory_text = f"Manual Reading - Mode: {pred_label}. Stats: HR {data.hr_mean}, EDA {data.eda_mean}. AI Advice: {advice}"
                    add_to_memory(memory_text)
                    
                except Exception as llm_error:
                    print(f"LLM Error: {llm_error}")
                    advice = f"Error generating advice: {str(llm_error)}"
        
        return {
            "prediction": pred_label,
            "confidence": round(confidence, 2),
            "advice": advice
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- RAG / Memory Functions ---
def get_embedding(text: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return None
    genai.configure(api_key=api_key)
    try:
        # Use embedding-001 or text-embedding-004
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
            title="Stress Log"
        )
        return result['embedding']
    except Exception as e:
        print(f"Embedding Error: {e}")
        return None

def add_to_memory(text: str):
    vec = get_embedding(text)
    if vec:
        vector = np.array([vec], dtype=np.float32)
        faiss_index.add(vector)
        stored_memories.append(text)
        print(f"Added to memory: {text[:50]}...")

def retrieve_memory(query: str, k: int = 3):
    if faiss_index.ntotal == 0: return ""
    
    vec = get_embedding(query)
    if vec:
        vector = np.array([vec], dtype=np.float32)
        D, I = faiss_index.search(vector, k)
        
        results = []
        for idx in I[0]:
            if idx != -1 and idx < len(stored_memories):
                results.append(stored_memories[idx])
        return "\n".join(results)
    return ""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

class ChatRequest(BaseModel):
    message: str
    context: str = ""
    history: List[Dict[str, str]] = []

@app.post("/chat")
async def chat_with_coach(data: ChatRequest):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="API Key not set.")
        
        genai.configure(api_key=api_key)
        # Use a system instruction if supported
        system_instruction = "You are an empathetic, expert AI Health Coach. Use the provided physiological context (Stress Level, Heart Rate, etc) to answer user questions. Be brief and supportive."
        
        model_llm = genai.GenerativeModel('models/gemini-flash-latest', system_instruction=system_instruction)
        
        # Convert history to Gemini format: [{'role': 'user'|'model', 'parts': ['...']}]
        gemini_history = []
        for msg in data.history:
            role = "user" if msg['role'] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg['content']]})
            
        chat = model_llm.start_chat(history=gemini_history)
        
        # Retrieve Long-Term Memory
        retrieved_context = retrieve_memory(data.message, k=2)
        
        # Prepend context to the user message invisibly to steer the model
        full_message = f"""
        [Relevant Past Memories]
        {retrieved_context}

        [Current User Context]
        {data.context}
        
        [User Question]
        {data.message}
        """
        
        response = chat.send_message(full_message)
        
        # Safe extraction
        ai_text = "I'm sorry, I couldn't generate a response."
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                ai_text = candidate.content.parts[0].text
            else:
                print(f"Empty Candidate. Finish Reason: {candidate.finish_reason}")
                if candidate.finish_reason == 3: # SAFETY
                    ai_text = "I simply cannot answer that due to safety guidelines."
                else:
                    ai_text = "I didn't have anything to say to that."
        
        return {"response": ai_text}

    except google.api_core.exceptions.ResourceExhausted:
        return {"response": "❄️ The AI is cooling down (Rate Limit Reached). Please wait about a minute before trying again."}
    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
