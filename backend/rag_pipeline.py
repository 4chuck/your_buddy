import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


class RAGPipeline:
    def __init__(self, collection_name="document_collection"):
        """
        Lightweight + Render-safe RAG pipeline
        """

        #  Chroma built-in embedding function (NO sentence-transformers needed)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        #  In-memory / ephemeral DB (Render-safe)
        self.client = chromadb.Client(
            Settings(anonymized_telemetry=False)
        )

        #  IMPORTANT: attach embedding function here
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    # ----------------------------
    # ADD DOCUMENTS
    # ----------------------------
    def add_documents(self, chunks):
        if not chunks:
            return

        documents = []
        metadatas = []
        ids = []

        for i, c in enumerate(chunks):
            # safe extraction
            if isinstance(c, dict):
                content = c.get("content", "")
                metadata = c.get("metadata", {})
            else:
                content = str(c)
                metadata = {}

            if content.strip():
                documents.append(content)
                metadatas.append(metadata)
                ids.append(f"doc_{len(ids)}")

        if not documents:
            return

        # ❌ NO MANUAL EMBEDDINGS (Chroma handles it internally)
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    # ----------------------------
    # QUERY
    # ----------------------------
    def query(self, query_text, n_results=5):
        if not query_text:
            return {"documents": [[]]}

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        return results

    # ----------------------------
    # RESET COLLECTION
    # ----------------------------
    def clear_collection(self):
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            print("Clear collection error:", e)