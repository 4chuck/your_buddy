import os
import sys


def main() -> None:
    # Force offline mode (no Firestore).
    os.environ.pop("FIREBASE_CREDENTIALS", None)
    os.environ["LOCAL_RAG_DB_PATH"] = os.path.join(
        os.path.dirname(__file__),
        "local_rag_test.sqlite3",
    )
    os.environ["RAG_FORCE_LOCAL"] = "1"

    repo_root = os.path.dirname(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from backend.rag_pipeline import RAGPipeline

    session_id = "local-test"
    rag = RAGPipeline()
    rag.clear_collection(session_id=session_id)

    rag.add_documents(
        chunks=[
            {"content": "FastAPI upload endpoint stores chunks locally.", "metadata": {"page": 1}},
            {"content": "Query should find relevant chunks without Firestore.", "metadata": {"page": 2}},
        ],
        user_id=session_id,
        session_id=session_id,
        source_file="test.txt",
    )

    res = rag.query("upload endpoint", n_results=3, user_id=session_id, session_id=session_id)
    print(res)


if __name__ == "__main__":
    main()
