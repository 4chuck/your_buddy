import os
import uuid
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from typing import cast
from chromadb.api.types import EmbeddingFunction

class RAGPipeline:
    def __init__(self, collection_name="document_collection"):

        self.embedding_function = cast(EmbeddingFunction,embedding_functions.DefaultEmbeddingFunction())

        # ✅ Persistent DB (important for Render)
        persist_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")

        self.client = chromadb.Client(
            Settings(
                persist_directory=persist_dir,
                anonymized_telemetry=False
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def add_documents(self, chunks):
        if not chunks:
            return

        documents = []
        metadatas = []
        ids = []

        for c in chunks:
            content = c.get("content", "") if isinstance(c, dict) else str(c)
            metadata = c.get("metadata", {}) if isinstance(c, dict) else {}

            if content.strip():
                documents.append(content)
                metadatas.append(metadata)

                # ✅ FIX: unique ID (no duplicates ever)
                ids.append(str(uuid.uuid4()))

        if not documents:
            return

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_text, n_results=5):
        if not query_text:
            return {"documents": [[]]}

        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

    def clear_collection(self):
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                embedding_function=self.embedding_function
            )
        except Exception as e:
            print("Clear collection error:", e)