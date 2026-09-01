# Vivi's Private AI

A 100% locally-hosted, private AI assistant powered by Llama 3.2, Stable Diffusion XL Turbo, and FastAPI. 
All data stays on your machine. No cloud subscriptions. No tracking.

## Features (v0.3 Prototype)
* **Local LLM Inference:** Powered by Ollama and Llama 3.2 (3B).
* **Advanced RAG (Retrieval-Augmented Generation):** Upload PDFs, DOCX, CSVs, and TXT files. The AI reads them, creates vector embeddings using `nomic-embed-text`, and stores them locally in ChromaDB.
* **Semantic Multi-user Memory:** Postgres database tracks individual users via browser `session_id`, isolating conversational memory privately.
* **Local Image Generation:** Integrated HuggingFace `diffusers` with `sdxl-turbo` running natively on Apple Silicon (MPS). The LLM autonomously issues JSON tool calls to trigger the image generator.
* **ChatGPT-style UI:** A clean, dark-mode `#0B0C0C` UI built with HTML/Tailwind CSS and Javascript fetch requests.

## Architecture
* **Frontend:** HTML, Tailwind CSS via CDN, Vanilla JS
* **Backend:** Python FastAPI, Uvicorn
* **Database:** PostgreSQL (Memory), ChromaDB (Vectors)
* **AI Models:** Llama 3.2, SDXL-Turbo, Nomic Embed Text

## Setup Instructions

### 1. Prerequisites
* Install [Ollama](https://ollama.ai/)
* Install PostgreSQL
* Mac with Apple Silicon (M1/M2/M3) recommended for MPS acceleration

### 2. Install Models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. Setup Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*(Note: requires `diffusers`, `transformers`, `fastapi`, `uvicorn`, `psycopg2`, `langchain`, `chromadb`, and `torch` with MPS support)*

### 4. Setup Database
```bash
createdb vivi_ai_memory
psql -d vivi_ai_memory -c "CREATE TABLE messages (id SERIAL PRIMARY KEY, role VARCHAR(50), content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, session_id VARCHAR(255) DEFAULT 'default_session');"
```

### 5. Run the Server
```bash
./start.sh
```
Then open `http://localhost:8000` in your browser.

## Privacy Guarantee
This AI does not connect to OpenAI, Anthropic, or any external API. The only external connection made is a one-time download of the SDXL-Turbo model weights from HuggingFace upon the first image generation request. Everything else is computed entirely on localhost.
