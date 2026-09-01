from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import tempfile
import uuid
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

# --- Setup ChromaDB & Embeddings ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Use Ollama for fully local embeddings!
class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            res = requests.post("http://localhost:11434/api/embeddings", json={
                "model": "nomic-embed-text",
                "prompt": text
            })
            embeddings.append(res.json().get("embedding", []))
        return embeddings

collection = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=OllamaEmbeddingFunction()
)

# --- FastAPI App ---
app = FastAPI(title="V's Private AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from datetime import datetime

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

pipeline = None
def get_image_pipeline():
    global pipeline
    if pipeline is None:
        print("Loading Stable Diffusion into Mac unified memory (this may take a moment)...")
        from diffusers import AutoPipelineForText2Image
        import torch
        pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        pipeline.to("mps")
    return pipeline

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_session"

def get_db_connection():
    return psycopg2.connect(
        dbname="vivi_ai_memory",
        user="vividhan",
        host="localhost"
    )

@app.get("/")
def read_root():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Welcome to V's Private AI API!"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Save file temporarily
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Extract Text based on file type
        suffix = suffix.lower()
        if suffix == '.pdf':
            loader = PyPDFLoader(tmp_path)
        elif suffix == '.docx':
            loader = Docx2txtLoader(tmp_path)
        elif suffix == '.csv':
            loader = CSVLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)
            
        docs = loader.load()

        # Chunk Text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        if not chunks:
            os.unlink(tmp_path)
            return {"status": "error", "error": "No text could be extracted from this document. If it is an image/scan, it requires OCR."}

        # Store in ChromaDB
        ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
        texts = [chunk.page_content for chunk in chunks]
        
        # Add to Vector DB (Embeddings are auto-generated via Ollama)
        collection.add(documents=texts, ids=ids)

        os.unlink(tmp_path)
        return {"status": "success", "message": f"Learned from {file.filename} ({len(chunks)} chunks)."}
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
def chat_with_ai(req: ChatRequest):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # 1. Search Vector DB for relevant context
        results = collection.query(query_texts=[req.message], n_results=3)
        context_str = ""
        if results['documents'] and len(results['documents'][0]) > 0:
            context_str = "\n\nRelevant Context from Documents:\n" + "\n".join(results['documents'][0])

        # 2. Save User Message
        cur.execute("INSERT INTO messages (role, content, session_id) VALUES (%s, %s, %s)", ('user', req.message, req.session_id))
        conn.commit()

        # 3. Fetch Conversation History for THIS session
        cur.execute("SELECT role, content FROM messages WHERE session_id = %s ORDER BY id ASC LIMIT 20", (req.session_id,))
        history = cur.fetchall()

        messages_for_ai = []
        
        # Inject System Prompt with Context AND Current Date
        current_date = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
        system_prompt = f"You are V's Private AI. The current date and time is {current_date}. "
        system_prompt += "If the user asks you to generate, draw, or create an image/picture, you MUST respond EXACTLY with a JSON object in this format: {\"action\": \"generate_image\", \"prompt\": \"<detailed physical description>\"}. Do not write any other text. "
        if context_str:
            system_prompt += "Use the provided context to answer questions. If the answer is not in the context, use your general knowledge." + context_str
            
        messages_for_ai.append({"role": "system", "content": system_prompt})

        for row in history:
            messages_for_ai.append({"role": row['role'], "content": row['content']})

        # 4. Call Ollama
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2",
                "messages": messages_for_ai,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            ai_text = response.json().get("message", {}).get("content", "").strip()
            
            # Check for Image Generation Tool Call
            if "generate_image" in ai_text:
                try:
                    clean_text = ai_text.replace("```json", "").replace("```", "").strip()
                    tool_data = json.loads(clean_text)
                    if tool_data.get("action") == "generate_image":
                        image_prompt = tool_data.get("prompt")
                        
                        # Generate Image (Lazy Loads Model on First Run!)
                        pipe = get_image_pipeline()
                        image = pipe(prompt=image_prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
                        
                        # Save Image
                        filename = f"{uuid.uuid4()}.png"
                        filepath = os.path.join(os.path.dirname(__file__), "static", filename)
                        image.save(filepath)
                        
                        ai_text = f"I generated the image for you based on the prompt: *{image_prompt}*\n\n![Generated Image](/static/{filename})"
                except Exception as e:
                    print("Image generation error:", e)
                    
            # Save AI Response
            cur.execute("INSERT INTO messages (role, content, session_id) VALUES (%s, %s, %s)", ('assistant', ai_text, req.session_id))
            conn.commit()
            return {"response": ai_text}
        else:
            return {"error": "AI Engine returned an error."}
            
    except Exception as e:
        return {"error": f"Backend Error: {str(e)}"}
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()

@app.post("/clear")
def clear_memory(req: dict):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        session_id = req.get("session_id", "default_session")
        cur.execute("DELETE FROM messages WHERE session_id = %s;", (session_id,))
        conn.commit()
        return {"status": "Memory cleared!"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if 'cur' in locals(): cur.close()
        if 'conn' in locals(): conn.close()
