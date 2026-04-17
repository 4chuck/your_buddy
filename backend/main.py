from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from uuid import uuid4
import os
import shutil

from utils import extract_text_from_file, chunk_text
from rag_pipeline import RAGPipeline
from agent import AIAgent


app = FastAPI(title="NotebookLM Clone API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGPipeline()
agent = AIAgent()

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    query: str
    mode: str = "qa"
    options: dict = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    all_chunks = []
    total = 0

    try:
        for file in files:
            path = os.path.join(UPLOAD_DIR, f"{uuid4().hex}_{file.filename}")

            with open(path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            text = extract_text_from_file(path)
            os.remove(path)

            if not text:
                continue

            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            total += len(chunks)

        if all_chunks:
            rag.add_documents(all_chunks)

        return {"status": "success", "chunks": total}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query(req: QueryRequest):
    results = rag.query(req.query, n_results=5)

    docs = results.get("documents", [[]])[0]
    context = "\n\n".join(docs)

    if not context:
        return {"response": "No data found"}

    if req.mode == "quiz":
        return {"response": agent.generate_quiz(context, req.options.get("num_questions", 5))}

    if req.mode == "simplify":
        return {"response": agent.explain_simply(context)}

    if req.mode == "agent":
        return {"response": agent.handle_agent_task(req.query, context)}

    return {"response": agent.ask_question(req.query, context)}