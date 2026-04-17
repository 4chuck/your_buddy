from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
import io

from utils import extract_text_from_file, chunk_text
from rag_pipeline import RAGPipeline
from agent import AIAgent

# ---------------- APP ----------------
app = FastAPI(title="NotebookLM Clone API")

# ---------------- CORS (PRODUCTION SAFE) ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://4chuck.github.io",
        "https://your-buddy-phi.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- INIT ----------------
rag = RAGPipeline()
agent = AIAgent()

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
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        all_chunks = []
        total_chunks = 0

        for file in files:

            # skip invalid files safely
            if not is_valid_file(file.filename):
                continue

            # read file safely (Render compatible)
            file_bytes = await file.read()
            file_stream = io.BytesIO(file_bytes)

            # extract text
            text_data = extract_text_from_file(file.filename, file_stream)

            if not text_data:
                continue

            # chunking
            chunks = chunk_text(text_data)

            if isinstance(chunks, list) and chunks:
                all_chunks.extend(chunks)
                total_chunks += len(chunks)

        # store in vector DB
        if all_chunks:
            rag.add_documents(all_chunks)

        return {
            "status": "success",
            "files_received": len(files),
            "chunks_created": total_chunks
        }

    except Exception as e:
        # IMPORTANT: prevents Render 502 crash loops
        return {
            "status": "error",
            "message": str(e)
        }

# ---------------- QUERY ----------------
@app.post("/query")
async def query(req: QueryRequest):
    try:
        results = rag.query(req.query, n_results=5)

        docs = results.get("documents", [])

        # ---------------- SAFE FLATTENING ----------------
        flat_docs = []

        if isinstance(docs, list):
            for d in docs:
                if isinstance(d, list):
                    flat_docs.extend(d)
                elif isinstance(d, str):
                    flat_docs.append(d)

        # clean context safely
        context_parts = [
            str(d).strip()
            for d in flat_docs
            if isinstance(d, str) and str(d).strip()
        ]

        context = "\n\n".join(context_parts)

        if not context:
            return {"response": "No relevant context found in uploaded documents."}

        # ---------------- MODES ----------------
        if req.mode == "quiz":
            return {
                "response": agent.generate_quiz(
                    context,
                    req.options.get("num_questions", 5)
                )
            }

        if req.mode == "simplify":
            return {"response": agent.explain_simply(context)}

        if req.mode == "agent":
            return {"response": agent.handle_agent_task(req.query, context)}

        return {"response": agent.ask_question(req.query, context)}

    except Exception as e:
        # NEVER crash server → prevents 502 + CORS confusion
        return {
            "response": f"Server error: {str(e)}"
        }