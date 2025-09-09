# app.py
import os
import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Load .env in dev (GROQ_API_KEY, LLM_MODEL, etc.)
load_dotenv()

# --- RAG index config (built by build_index.py) ---
VEC_PATH = "index/greene_vectors.npy"
META_PATH = "index/greene_metadata.jsonl"

# ---- MiniLM embedder (local, free) ----
# keep this tiny so we don't pull sentence-transformers at import time
_embedder_model = None
def _get_embedder():
    global _embedder_model
    if _embedder_model is None:
        from sentence_transformers import SentenceTransformer  # lazy import
        name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _embedder_model = SentenceTransformer(name)
    return _embedder_model

def embed_texts(texts):
    m = _get_embedder()
    v = m.encode(texts, normalize_embeddings=True)
    return np.asarray(v, dtype="float32")

# ---- Lazy index loader ----
_vectors = None
_meta = None
def load_index():
    """Loads vectors + metadata once; reuse for all requests."""
    global _vectors, _meta
    if _vectors is None:
        _vectors = np.load(VEC_PATH).astype("float32")        # shape: (N, 384)
        _meta = [json.loads(l) for l in open(META_PATH, encoding="utf-8")]
    return _vectors, _meta

def search_topk(query: str, k: int = 4):
    X, M = load_index()
    qv = embed_texts([query])[0]                              # (384,), unit-normalized
    scores = X @ qv                                           # cosine similarity via dot product
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i]), M[i])] if k == 1 else [(int(i), float(scores[i]), M[i]) for i in idx]

# ---- FastAPI setup ----
app = FastAPI(title="Greene Mentor API")

class AskReq(BaseModel):
    query: str

@app.get("/")
def home():
    return {"ok": True, "app": "Greene Mentor API", "try": "/docs, POST /ask, POST /ask_rag"}

# ---- Plain generation (Groq) ----
@app.post("/ask")
def ask(req: AskReq):
    """
    Style-only: answers in a Robert-Greene-inspired tone (no retrieval).
    """
    prompt = (
        "Act as a strategic, historically-aware mentor inspired by Robert Greene.\n"
        "Follow this structure:\n"
        "1) Diagnosis (one paragraph)\n"
        "2) Moves (3 bullet points, each actionable)\n"
        "3) Closing principle (one sentence)\n\n"
        f"User: {req.query}\nAssistant:"
    )

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return {"error": "GROQ_API_KEY missing in .env"}

    client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=700,
    )
    return {"answer": resp.choices[0].message.content, "provider": "groq", "model": model}

# ---- RAG generation (Groq + local corpus) ----
@app.post("/ask_rag")
def ask_rag(req: AskReq, k: int = 4):
    """
    Retrieval-augmented: finds top-k chunks from your transcripts and cites them.
    Requires index/greene_vectors.npy and index/greene_metadata.jsonl.
    """
    try:
        hits = search_topk(req.query, k=k)
    except FileNotFoundError:
        return {"error": "RAG index not found. Run build_index.py first."}

    # Build context block with inline tags
    blocks = []
    for i, (_, score, m) in enumerate(hits, 1):
        blocks.append(f"[S{i}] {m['title']} — {m['source']}\n{m['content']}")
    context = "\n\n".join(blocks)

    prompt = (
        "You are MentorRG, an AI mentor inspired by Robert Greene’s ideas (not the author).\n"
        "Use the provided context and cite with [S1], [S2], etc.\n"
        "Structure: 1) Diagnosis (brief) 2) Moves (3 bullets, actionable) 3) Closing principle (1 line)\n\n"
        f"Question: {req.query}\n\nContext:\n{context}\n\nAnswer:"
    )

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return {"error": "GROQ_API_KEY missing in .env"}

    client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
    model = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.45,
        max_tokens=700,
    )

    return {
        "answer": resp.choices[0].message.content,
        "sources": [{"tag": f"S{i+1}", "title": h[2]["title"], "source": h[2]["source"]} for i, h in enumerate(hits)],
        "k": k,
    }
