import os
import re
import uuid
import logging
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from backend.firebase_config import db
except ImportError:
    from firebase_config import db

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, collection_name: str = "document_chunks"):
        self.collection_name = collection_name
        self.max_chunk_chars = int(os.getenv("RAG_MAX_CHUNK_CHARS", "4000"))
        self.max_candidates = int(os.getenv("RAG_MAX_CANDIDATES", "200"))
        self.firestore_timeout_s = float(os.getenv("FIRESTORE_TIMEOUT_SECONDS", "10"))
        self.force_local = (os.getenv("RAG_FORCE_LOCAL", "0") or "").strip().lower() in {"1", "true", "yes"}
        self.local_db_path = os.getenv(
            "LOCAL_RAG_DB_PATH",
            os.path.join(os.path.dirname(__file__), "local_rag.sqlite3"),
        )

        # If Firestore isn't initialized (missing credentials, init error, etc.),
        # fall back to local SQLite storage automatically.
        self.firestore_enabled = (db is not None) and (not self.force_local)

        self._init_local_store()

    def _init_local_store(self) -> None:
        os.makedirs(os.path.dirname(self.local_db_path), exist_ok=True)
        with sqlite3.connect(self.local_db_path, timeout=30) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_id TEXT,
                    content TEXT,
                    chunk_index INTEGER,
                    source_file TEXT,
                    created_at REAL,
                    metadata_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_document_chunks_session ON document_chunks(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_document_chunks_user ON document_chunks(user_id)"
            )

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())

    def _safe_content(self, doc: Any) -> str:
        if not isinstance(doc, dict):
            return ""
        return str(doc.get("content") or "").strip()

    def _safe_created_at(self, doc: Any) -> float:
        if not isinstance(doc, dict):
            return 0.0
        created_at = doc.get("created_at")
        if isinstance(created_at, datetime):
            try:
                return created_at.timestamp()
            except Exception:
                return 0.0
        if isinstance(created_at, (int, float)):
            try:
                return float(created_at)
            except Exception:
                return 0.0
        return 0.0

    def _score(self, query_tokens: List[str], content: str, lower_query: str) -> float:
        content_text = (content or "").strip()
        if not content_text:
            return 0.0

        content_tokens = self._tokenize(content_text)
        if not content_tokens or not query_tokens:
            return 0.0

        q = Counter(query_tokens)
        c = Counter(content_tokens)
        overlap = sum(min(q[t], c.get(t, 0)) for t in q)

        content_lower = content_text.lower()
        phrase_tokens = query_tokens[:6]
        phrase = " ".join(phrase_tokens).strip()
        phrase_bonus = 1.5 if phrase and phrase in content_lower else 0.0
        exact_bonus = 2.0 if lower_query and lower_query in content_lower else 0.0

        return float(overlap) + phrase_bonus + exact_bonus

    def add_documents(
        self,
        chunks: List[Any],
        user_id: str,
        session_id: str,
        source_file: Optional[str] = None,
    ) -> None:
        if not chunks:
            return
        if not session_id or not str(session_id).strip():
            return

        # Prefer Firestore when available; automatically fall back to local SQLite
        # if Firestore is unavailable or the write fails (offline dev, emulator not
        # running, network restrictions, etc.).
        if self.firestore_enabled:
            try:
                self._add_documents_firestore(
                    chunks=chunks,
                    user_id=user_id,
                    session_id=session_id,
                    source_file=source_file,
                )
                return
            except Exception as e:
                logger.warning("Firestore add_documents failed; falling back to SQLite: %s", e)
                self.firestore_enabled = False

        self._add_documents_sqlite(
            chunks=chunks,
            user_id=user_id,
            session_id=session_id,
            source_file=source_file,
        )

    def _add_documents_firestore(
        self,
        chunks: List[Any],
        user_id: str,
        session_id: str,
        source_file: Optional[str],
    ) -> None:
        if db is None:
            raise RuntimeError("Firestore client not initialized")

        collection = db.collection(self.collection_name)
        batch = db.batch()
        pending = 0

        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                content = str(chunk.get("content") or "").strip()
                metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else None
            else:
                content = str(chunk).strip()
                metadata = None

            if not content:
                continue

            doc: Dict[str, Any] = {
                "chunk_id": str(uuid.uuid4()),
                "user_id": str(user_id or session_id),
                "session_id": str(session_id),
                "content": content[: self.max_chunk_chars],
                "chunk_index": i,
                "source_file": source_file or "upload",
                "created_at": datetime.now(timezone.utc),
            }
            if metadata:
                doc["metadata"] = metadata

            doc_ref = collection.document(doc["chunk_id"])
            batch.set(doc_ref, doc)
            pending += 1

            if pending >= 400:
                batch.commit(timeout=self.firestore_timeout_s)
                batch = db.batch()
                pending = 0

        if pending:
            batch.commit(timeout=self.firestore_timeout_s)

    def _add_documents_sqlite(
        self,
        chunks: List[Any],
        user_id: str,
        session_id: str,
        source_file: Optional[str],
    ) -> None:
        now = datetime.now(timezone.utc).timestamp()
        user_id_value = str(user_id or session_id)
        session_id_value = str(session_id)
        source_file_value = source_file or "upload"

        rows: List[Tuple[str, str, str, str, int, str, float, Optional[str]]] = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                content = str(chunk.get("content") or "").strip()
                metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else None
            else:
                content = str(chunk).strip()
                metadata = None

            if not content:
                continue

            metadata_json = json.dumps(metadata) if isinstance(metadata, dict) else None
            rows.append(
                (
                    str(uuid.uuid4()),
                    user_id_value,
                    session_id_value,
                    content[: self.max_chunk_chars],
                    int(i),
                    source_file_value,
                    float(now),
                    metadata_json,
                )
            )

        if not rows:
            return

        with sqlite3.connect(self.local_db_path, timeout=30) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO document_chunks
                    (chunk_id, user_id, session_id, content, chunk_index, source_file, created_at, metadata_json)
                VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def _fetch_session_docs(self, session_id: str) -> List[Dict[str, Any]]:
        try:
            if db is None:
                return []

            col = db.collection(self.collection_name)
            docs = (
                col.where("session_id", "==", str(session_id))
                .limit(self.max_candidates)
                .stream(timeout=self.firestore_timeout_s)
            )
            items: List[Dict[str, Any]] = []
            for d in docs:
                data = d.to_dict()
                if isinstance(data, dict):
                    items.append(data)
            return items
        except Exception as e:
            logger.error("fetch session docs error: %s", e)
            return []

    def _fetch_user_fallback_docs(self, user_id: str) -> List[Dict[str, Any]]:
        try:
            if db is None:
                return []

            col = db.collection(self.collection_name)
            docs = (
                col.where("user_id", "==", str(user_id))
                .limit(self.max_candidates)
                .stream(timeout=self.firestore_timeout_s)
            )
            items: List[Dict[str, Any]] = []
            for d in docs:
                data = d.to_dict()
                if isinstance(data, dict):
                    items.append(data)
            return items
        except Exception as e:
            logger.error("fetch user fallback docs error: %s", e)
            return []

    def _fetch_sqlite(self, session_id: str, user_id: Optional[str]) -> List[Dict[str, Any]]:
        session_id_value = str(session_id or "").strip()
        user_id_value = str(user_id or "").strip()
        if not session_id_value and not user_id_value:
            return []

        try:
            with sqlite3.connect(self.local_db_path, timeout=30) as conn:
                conn.row_factory = sqlite3.Row

                if session_id_value:
                    cur = conn.execute(
                        """
                        SELECT user_id, session_id, content, chunk_index, source_file, created_at, metadata_json
                        FROM document_chunks
                        WHERE session_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                        """,
                        (session_id_value, int(self.max_candidates)),
                    )
                else:
                    cur = conn.execute(
                        """
                        SELECT user_id, session_id, content, chunk_index, source_file, created_at, metadata_json
                        FROM document_chunks
                        WHERE user_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                        """,
                        (user_id_value, int(self.max_candidates)),
                    )

                items: List[Dict[str, Any]] = []
                for row in cur.fetchall():
                    d: Dict[str, Any] = dict(row)
                    metadata_json = d.pop("metadata_json", None)
                    if metadata_json:
                        try:
                            d["metadata"] = json.loads(metadata_json)
                        except Exception:
                            pass
                    items.append(d)
                return items
        except Exception as e:
            logger.error("fetch sqlite docs error: %s", e)
            return []

    def _fetch(self, session_id: str, user_id: Optional[str]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if self.firestore_enabled:
            items = self._fetch_session_docs(session_id)
        if items:
            return items

        if self.firestore_enabled and user_id and str(user_id).strip() and str(user_id) == str(session_id):
            items = self._fetch_user_fallback_docs(user_id)
            if items:
                return items

        # Offline/local fallback
        return self._fetch_sqlite(session_id=session_id, user_id=user_id)

    def _recent_contents(self, docs: List[Dict[str, Any]], n_results: int) -> List[str]:
        docs_sorted = sorted(docs, key=self._safe_created_at, reverse=True)
        out: List[str] = []
        for d in docs_sorted:
            content = self._safe_content(d)
            if content:
                out.append(content)
            if len(out) >= n_results:
                break
        return out

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, List[List[str]]]:
        scope_session_id = str(session_id or "").strip() or str(user_id or "").strip()
        if not scope_session_id:
            return {"documents": [[]]}

        docs = self._fetch(scope_session_id, user_id=user_id)
        if not docs:
            return {"documents": [[]]}

        if not query_text or not str(query_text).strip():
            return {"documents": [self._recent_contents(docs, n_results)]}

        query_tokens = self._tokenize(query_text)
        lower_query = str(query_text).lower().strip()

        scored: List[Tuple[float, str]] = []
        for d in docs:
            content = self._safe_content(d)
            if not content:
                continue
            score = self._score(query_tokens, content, lower_query=lower_query)
            if score > 0:
                scored.append((score, content))

        if not scored:
            return {"documents": [[]]}

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [content for _, content in scored[:n_results]]
        return {"documents": [top]}

    def clear_collection(self, session_id: Optional[str] = None) -> int:
        count = 0

        if self.firestore_enabled and db is not None:
            try:
                col = db.collection(self.collection_name)
                docs = (
                    col.where("session_id", "==", str(session_id)).stream()
                    if session_id
                    else col.stream()
                )

                batch = db.batch()
                for d in docs:
                    batch.delete(d.reference)
                    count += 1
                    if count % 400 == 0:
                        batch.commit()
                        batch = db.batch()

                if count % 400:
                    batch.commit()
            except Exception as e:
                logger.warning("Firestore clear_collection failed; continuing with SQLite: %s", e)
                self.firestore_enabled = False

        # Always clear local fallback as well.
        try:
            with sqlite3.connect(self.local_db_path, timeout=30) as conn:
                if session_id:
                    cur = conn.execute(
                        "DELETE FROM document_chunks WHERE session_id = ?",
                        (str(session_id),),
                    )
                else:
                    cur = conn.execute("DELETE FROM document_chunks")
                count += int(cur.rowcount or 0)
        except Exception as e:
            logger.error("SQLite clear_collection failed: %s", e)

        return count
