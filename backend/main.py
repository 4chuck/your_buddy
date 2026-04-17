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

# ----------------------------
# CORS CONFIG (Frontend support)
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# INIT PIPELINES
# ----------------------------
rag = RAGPipeline()
agent = AIAgent()

# ----------------------------
# TEMP UPLOAD DIRECTORY
# ----------------------------
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------
# REQUEST MODEL
# ----------------------------
class QueryRequest(BaseModel):
    query: str
    mode: str = "qa"  # qa | quiz | simplify | agent
    options: dict = {}

# ----------------------------
# HEALTH CHECK
# ----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ----------------------------
# UPLOAD PDF / FILES
# ----------------------------
@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    total_chunks = 0
    all_chunks = []

    try:
        for upload_file in files:
            unique_name = f"{uuid4().hex}_{upload_file.filename}"
            file_path = os.path.join(UPLOAD_DIR, unique_name)

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)

            print(f"Extracting: {upload_file.filename}")

            text_data = extract_text_from_file(file_path)
            os.remove(file_path)

            if not text_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not extract text from {upload_file.filename}",
                )

            chunks = chunk_text(text_data)
            all_chunks.extend(chunks)
            total_chunks += len(chunks)

        if all_chunks:
            print(f"Adding {len(all_chunks)} chunks to vector DB...")
            rag.add_documents(all_chunks)

        return {
            "status": "success",
            "message": f"Processed {len(files)} files with {total_chunks} chunks"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------
# QUERY ENDPOINT (RAG + AGENT)
# ----------------------------
@app.post("/query")
async def process_query(request: QueryRequest):
    query = request.query
    mode = request.mode

    try:
        # Retrieve context from vector DB
        results = rag.query(query, n_results=5)

        context_chunks = results.get("documents", [[]])[0]
        context = "\n\n".join(context_chunks)

        if not context:
            return {
                "status": "success",
                "response": "No relevant information found in uploaded documents."
            }

        # Route based on mode
        if mode == "qa":
            response = agent.ask_question(query, context)

        elif mode == "quiz":
            num_q = request.options.get("num_questions", 5)
            response = agent.generate_quiz(context, num_questions=num_q)

        elif mode == "simplify":
            response = agent.explain_simply(context)

        elif mode == "agent":
            response = agent.handle_agent_task(query, context)

        else:
            response = agent.ask_question(query, context)

        return {
            "status": "success",
            "response": response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))