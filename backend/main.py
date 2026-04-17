from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from uuid import uuid4
import os
import shutil

from backend.utils import extract_text_from_file, chunk_text
from backend.rag_pipeline import RAGPipeline
from backend.agent import AIAgent

app = FastAPI(title="NotebookLM Clone API")

# Setup CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipelines
rag = RAGPipeline()
agent = AIAgent()

# Temporary upload folder
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str
    mode: str = "qa"  # qa | quiz | simplify | agent
    options: dict = {}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    total_chunks = 0
    try:
        all_chunks = []
        for upload_file in files:
            unique_name = f"{uuid4().hex}_{upload_file.filename}"
            file_path = os.path.join(UPLOAD_DIR, unique_name)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)

            print(f"Extracting text from {upload_file.filename}...")
            text_data = extract_text_from_file(file_path)
            os.remove(file_path)

            if not text_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not extract text from {upload_file.filename}. Supported formats include PDF, DOCX, PPTX, TXT, MD, CSV, JSON, HTML, XML and other text files.",
                )

            chunks = chunk_text(text_data)
            all_chunks.extend(chunks)
            total_chunks += len(chunks)

        if all_chunks:
            print(f"Adding {len(all_chunks)} chunks to vector DB...")
            rag.add_documents(all_chunks)

        return {
            "status": "success",
            "message": f"Successfully processed {len(files)} files with {total_chunks} chunks.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/query")
async def process_query(request: QueryRequest):
    query = request.query
    mode = request.mode
    
    # 1. Retrieve context chunks from Vector DB
    try:
        search_query = query
        # If the user is asking for a quiz on chapter X, we search for chapter X.
        # If it's pure agent task, we might just search what they requested.
        results = rag.query(search_query, n_results=5)
        
        context_chunks = results['documents'][0] if results['documents'] else []
        context = "\n\n".join(context_chunks)
        
        if not context:
            return {"response": "I couldn't find relevant information in the uploaded PDF. Please try a different query."}
            
        # 2. Route to appropriate agent function based on mode
        response = ""
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
            
        return {"status": "success", "response": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
