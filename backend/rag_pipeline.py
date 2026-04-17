import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import os

class RAGPipeline:
    def __init__(self, collection_name="document_collection"):
        # Initialize the embedding function
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB client (persistent storage)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks):
        """Adds chunks to the vector database."""
        documents = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [f"id_{i}" for i in range(len(chunks))]
        
        # In a real app, you'd generate embeddings first or let Chroma handle it
        # We'll generate them manually for better control
        embeddings = self.embedding_model.encode(documents).tolist()
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_text, n_results=5):
        """Retrieves top-k relevant chunks."""
        query_embeddings = self.embedding_model.encode([query_text]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )
        return results

    def clear_collection(self):
        """Web demo: clear collection for new uploads."""
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.get_or_create_collection(name=self.collection.name)
        except Exception as e:
            print(f"Error clearing collection: {e}")
