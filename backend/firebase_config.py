import os
from pathlib import Path
from typing import Optional

import firebase_admin
from firebase_admin import credentials, firestore


def _default_key_path() -> Optional[str]:
    # Prefer explicit env var, otherwise try the repo-local dev key if present.
    env_path = (os.getenv("FIREBASE_CREDENTIALS") or "").strip()
    if env_path:
        return env_path

    local = Path(__file__).resolve().parent / "serviceAccountKey.json"
    if local.exists():
        return str(local)

    return None


def get_db():
    """
    Best-effort Firestore client initializer.

    Returns `None` when credentials are missing/invalid so the app can fall back
    to a local store (useful for offline dev).
    """
    try:
        key_path = _default_key_path()
        if not key_path:
            return None

        # Prevent re-initialization (important in FastAPI reloads)
        if not firebase_admin._apps:
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)

        return firestore.client()
    except Exception:
        return None


# Backwards-compatible export used across the codebase.
db = get_db()
