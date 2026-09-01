#!/bin/bash
echo "🤖 Starting V's Private AI Backend..."
cd "$(dirname "$0")"

# Ensure Ollama is running
echo "🧠 Checking AI Engine (Ollama)..."
if ! pgrep -x "ollama" > /dev/null
then
    echo "Starting Ollama service..."
    OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve &
    sleep 2
else
    echo "✅ Ollama is already running."
fi

# Start the FastAPI server
echo "🚀 Starting API Server on http://localhost:8000..."
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
