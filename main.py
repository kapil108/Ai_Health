from src.api import app

if __name__ == "__main__":
    import uvicorn
    # Use port 8001 by default to avoid conflicts
    uvicorn.run(app, host="127.0.0.1", port=8001)
