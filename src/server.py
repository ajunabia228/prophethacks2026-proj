"""
Runs a FastAPI server that exposes an endpoint for forecasting probabilities of event outcomes using an OpenAI model. 
The server receives event details, constructs a prompt, and returns calibrated probability estimates for each outcome in JSON format.

how to run:
1. Create a virtual environment and install dependencies: `pip install fastapi uvicorn openai python-dotenv`
2. Set your OpenRouter API key in a .env file: `OPENROUTER_API_KEY=sk-or-v1-...`
3. Start the server: `python server.py`

Usage:
    prophet forecast predict --events events.json --agent-url http://localhost:8000/predict -o predictions.json


For Render:
    1. Create a Render web service with the following settings:
    - Environment: Python 3
    - Build Command: `pip install -r requirements.txt`
    - Start Command: `cd src && uvicorn server:app --host 0.0.0.0 --port $PORT`
    2. Set the OPENROUTER_API_KEY environment variable in Render's dashboard.

"""


import os
from dotenv import load_dotenv
from fastapi import FastAPI

app = FastAPI(title="ProphetHacks 2026 Forecast Agent")

# Fix for ENVs not loading when not using our custom CLI setup
# (make sure you have a .env file with OPENROUTER_API_KEY with your working director [ie: vscode workspace])
load_dotenv()
if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY not found in environment variables. Please set it in a .env file or your environment.")

# necessary as envs must be loaded before creating the OpenAI client in my_agent.py
from my_agent import predict

@app.post("/predict")
async def predict_endpoint(event: dict):
    """Calls the predict function from my_agent.py"""
    return predict(event)

# Just a simple health check endpoint to verify the server is running
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.head("/ping")
async def ping_head():
    return

# Run the server with: uvicorn server:app --host
def main() -> None:
    import uvicorn

    port = int(os.getenv("LOCALHOST_PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)

if __name__ == "__main__":
    main()