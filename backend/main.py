from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import io
import logging

# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from utils import extract_text_from_file, chunk_text
from rag_pipeline import RAGPipeline
from agent import AIAgent

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------- APP ----------------
app = FastAPI(title="NotebookLM Clone API")

# ---------------- RATE LIMITER ----------------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "status": "error",
            "message": "Too many requests. Please slow down."
        },
    )

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://4chuck.github.io",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "https://your-buddy-phi.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- INIT ----------------
rag = RAGPipeline()
agent = AIAgent()

# ---------------- MEMORY ----------------
chat_memory: Dict[str, List[Dict[str, str]]] = {}

# ---------------- REQUEST MODEL ----------------
class QueryRequest(BaseModel):
    query: str
    mode: str = "qa"
    options: Dict[str, Any] = Field(default_factory=dict)

# ---------------- HEALTH ----------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Backend running"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- FILE VALIDATION ----------------
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".pptx"}

def is_valid_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- UPLOAD ----------------
@app.post("/upload")
@limiter.limit("10/minute")
async def upload_files(request: Request, files: List[UploadFile] = File(...)):
    try:
        all_chunks = []
        total_chunks = 0

        for file in files:
            if not file.filename or not is_valid_file(file.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file: {file.filename}"
                )

            file_bytes = await file.read()
            file_stream = io.BytesIO(file_bytes)

            text_data = extract_text_from_file(file.filename, file_stream)

            if not text_data:
                continue

            chunks = chunk_text(text_data)

            if isinstance(chunks, list) and chunks:
                all_chunks.extend(chunks)
                total_chunks += len(chunks)

        if all_chunks:
            rag.add_documents(all_chunks)

        logging.info(f"Uploaded {len(files)} files → {total_chunks} chunks")

        return {
            "status": "success",
            "files_received": len(files),
            "chunks_created": total_chunks
        }

    except Exception as e:
        logging.error(f"Upload error: {str(e)}")

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "File processing failed"
            }
        )

# ---------------- QUERY ----------------
@app.post("/query")
@limiter.limit("5/minute")
async def query(req: QueryRequest, request: Request):
    try:
        logging.info(f"Query received: {req.query}")

        # ---------------- SAFE SESSION ID (FIXED) ----------------
        session_id = (
            request.client.host
            if request.client and request.client.host
            else "unknown"
        )

        history = chat_memory.get(session_id, [])

        # ---------------- RAG ----------------
        results = rag.query(req.query, n_results=5)
        docs = results.get("documents", [])

        flat_docs = []

        if isinstance(docs, list):
            for d in docs:
                if isinstance(d, list):
                    flat_docs.extend(d)
                elif isinstance(d, str):
                    flat_docs.append(d)

        context_parts = [
            str(d).strip()
            for d in flat_docs
            if isinstance(d, str) and str(d).strip()
        ]

        context = "\n\n".join(context_parts)

        if not context:
            return {
                "status": "error",
                "message": "No relevant context found in uploaded documents."
            }

        # ---------------- AGENT MODES ----------------
        if req.mode == "quiz":
            response = agent.generate_quiz(
                context,
                req.options.get("num_questions", 5)
            )

        elif req.mode == "simplify":
            response = agent.explain_simply(context)

        elif req.mode == "agent":
            response = agent.handle_agent_task(req.query, context)

        else:
            response = agent.ask_question(req.query, context)

        # ---------------- SAVE MEMORY ----------------
        history.append({"role": "user", "content": req.query})
        history.append({"role": "assistant", "content": response})

        chat_memory[session_id] = history[-10:]

        # ---------------- SAFE RESPONSE ----------------
        return {
            "status": "success",
            "response": response
        }

    except Exception as e:
        logging.error(f"Query error: {str(e)}")

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
