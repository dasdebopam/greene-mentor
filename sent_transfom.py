from sentence_transformers import SentenceTransformer
import os, numpy as np
_model=None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(os.getenv("EMBEDDING_MODEL","all-MiniLM-L6-v2"))
    return _model
def embed_texts(texts):
    v = get_model().encode(texts, normalize_embeddings=True)
    return np.asarray(v, dtype="float32")
