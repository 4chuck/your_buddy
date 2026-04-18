from backend.firebase_config import db

collections = list(db.collections())
print("Connected to Firestore ✅")
print("Collections:", collections)
