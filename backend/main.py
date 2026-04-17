from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from uuid import uuid4
import os
import shutil

from utils import extract_text_from_file, chunk_text
from rag_pipeline import RAGPipeline
from agent import AIAgent

app = FastAPI(title="NotebookLM Clone API")

# ---------------- CORS ----------------
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

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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


# ---------------- FILE UPLOAD ----------------
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".pptx"}

def is_valid_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    all_chunks = []
    total_chunks = 0

    try:
        for file in files:

            if not is_valid_file(file.filename):
                continue

            path = os.path.join(UPLOAD_DIR, f"{uuid4().hex}_{file.filename}")

            with open(path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            file.file.close()

            text_data = extract_text_from_file(path)

            try:
                os.remove(path)
            except:
                pass

            if not text_data:
                continue

            chunks = chunk_text(text_data)

            if chunks:
                all_chunks.extend(chunks)
                total_chunks += len(chunks)

        if all_chunks:
            rag.add_documents(all_chunks)

        return {
            "status": "success",
            "files_processed": len(files),
            "chunks_created": total_chunks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

        context = "\n\n".join(
            str(d).strip() for d in flat_docs if isinstance(d, str) and d.strip()
        )

        if not context.strip():
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
        raise HTTPException(status_code=500, detail=str(e))