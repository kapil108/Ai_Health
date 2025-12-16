import sys
import os
import asyncio
from contextlib import asynccontextmanager

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

try:
    print("Attempting to import src.api...")
    from src.api import app, lifespan
    print("Successfully imported src.api")
except ImportError as e:
    print(f"CRITICAL: Failed to import src.api: {e}")
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL: Error during import of src.api: {e}")
    sys.exit(1)

async def test_startup():
    print("Testing application startup (lifespan)...")
    try:
        async with lifespan(app):
            print("Startup successful! Model and Gemini configuration loaded.")
    except Exception as e:
        print(f"CRITICAL: Startup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_startup())
