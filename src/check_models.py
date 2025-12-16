import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in environment.")
else:
    genai.configure(api_key=api_key)
    with open("models.txt", "w") as f:
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    f.write(f"{m.name}\n")
                    print(f"- Name: {m.name}")
        except Exception as e:
            f.write(f"Error: {e}")
            print(f"Error: {e}")
