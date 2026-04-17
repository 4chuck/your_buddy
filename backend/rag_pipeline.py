import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    def __init__(self, collection_name="document_collection"):
        """
        Lightweight + Render-safe RAG pipeline
        """

        # ⚠️ Small embedding model (OK for hackathon)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # ⚠️ In-memory ChromaDB (IMPORTANT for Render)
        self.client = chromadb.Client(
            Settings(
                anonymized_telemetry=False
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    # ----------------------------
    # ADD DOCUMENTS
    # ----------------------------
    def add_documents(self, chunks):
        if not chunks:
            return

        documents = []
        metadatas = []

        for i, c in enumerate(chunks):
            # SAFE extraction (prevents crash)
            if isinstance(c, dict):
                content = c.get("content", "")
                metadata = c.get("metadata", {})
            else:
                content = str(c)
                metadata = {}

            if content.strip():
                documents.append(content)
                metadatas.append(metadata)

        if not documents:
            return

        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()

        ids = [f"doc_{i}" for i in range(len(documents))]

        self.collection.add(
            embeddings=embeddings,
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

        query_embedding = self.embedding_model.encode([query_text]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        return results

    # ----------------------------
    # RESET COLLECTION (SAFE)
    # ----------------------------
    def clear_collection(self):
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name
            )
        except Exception as e:
            print("Clear collection error:", e)