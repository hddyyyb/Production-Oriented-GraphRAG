# src/ingest/embedder.py
from sentence_transformers import SentenceTransformer

def embed_text(text: str):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model.encode(text)
